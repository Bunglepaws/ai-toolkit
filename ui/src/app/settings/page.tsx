'use client';

import { useEffect, useState } from 'react';
import useSettings from '@/hooks/useSettings';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { CircleHelp } from 'lucide-react';
import { openDoc } from '@/components/DocModal';

const aiConfigCheckDoc = {
  title: 'AI Config Check',
  description: (
    <div className="space-y-3 text-sm">
      <p>
        The <strong>Check Config ✦</strong> button appears on each training job and sends your job configuration
        to an LLM for analysis. It assembles multiple data sources before making the call:
      </p>
      <ul className="list-disc ml-4 space-y-1">
        <li><strong>Job configuration</strong> — all training parameters (LR, rank, steps, optimizer, network type, etc.)</li>
        <li><strong>Dataset stats</strong> — image count and resolution bucket distribution (written to DB after caching). The LLM checks whether your configured training resolution matches your actual image dimensions, flags upscaling risk, uneven bucket distributions, and buckets with too few images to train stably</li>
        <li><strong>Loss curve</strong> — last 200 loss data points if the job has run before, so the LLM can spot divergence trends</li>
        <li><strong>System stats</strong> — GPU VRAM total/used/free and system RAM. Used to flag OOM risk based on your configured resolution (higher resolution = exponentially more VRAM), thermal throttling, and whether you have headroom to increase batch size or resolution</li>
        <li><strong>Sample images</strong> — recent generated sample images (or MP4 frames for video models) so it can assess output quality, white noise, mode collapse, oversaturation, or temporal inconsistency</li>
        <li><strong>Dataset images</strong> — a sample of your training images so it can check quantity, resolution fit, style consistency, and watermarks</li>
      </ul>
      <p>
        The LLM returns structured findings, each with a <strong>severity</strong> (info / warning / error),
        a <strong>confidence level</strong> (high = official docs, medium = community sources, low = inferred),
        and a <strong>references</strong> list showing where the advice comes from. High-confidence findings
        have an Apply button that patches the config field directly.
      </p>
      <p>
        Works with any OpenAI-compatible endpoint — Anthropic Claude, local Ollama, or any other provider.
        For visual sample analysis a vision-capable model is required (e.g. Qwen2.5-VL or Qwen3-VL via Ollama).
        Text-only config analysis works with any model.
      </p>
      <p className="text-gray-400">
        Web search (Ollama only) fetches up-to-date community guidance about your specific model architecture
        before the LLM call, which helps with newer models the LLM may not have training data on.
      </p>
    </div>
  ),
};

export default function Settings() {
  const { settings, setSettings } = useSettings();
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('saving');

    apiClient
      .post('/api/settings', settings)
      .then(() => {
        setStatus('success');
      })
      .catch(error => {
        console.error('Error saving settings:', error);
        setStatus('error');
      })
      .finally(() => {
        setTimeout(() => setStatus('idle'), 2000);
      });
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setSettings(prev => ({ ...prev, [name]: type === 'checkbox' ? (checked ? 'true' : 'false') : value }));
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Settings</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <div className="space-y-4">
                <div>
                  <label htmlFor="HF_TOKEN" className="block text-sm font-medium mb-2">
                    Hugging Face Token
                    <div className="text-gray-500 text-sm ml-1">
                      Create a Read token on{' '}
                      <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                        {' '}
                        Huggingface
                      </a>{' '}
                      if you need to access gated/private models.
                    </div>
                  </label>
                  <input
                    type="password"
                    id="HF_TOKEN"
                    name="HF_TOKEN"
                    value={settings.HF_TOKEN}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter your Hugging Face token"
                  />
                </div>

                <div>
                  <label htmlFor="GEMMA_API_KEY" className="block text-sm font-medium mb-2">
                    Lightricks Gemma Text Encoding API Key
                    <div className="text-gray-500 text-sm ml-1">
                      For LTX-2.3 training. Lets you encode text prompts via the cloud API instead
                      of loading the 12B Gemma model locally, saving ~24 GB of VRAM.{' '}
                      <a href="https://console.ltx.video" target="_blank" rel="noreferrer">
                        Get a free key at console.ltx.video
                      </a>
                      {' '}(sign up → API section).
                    </div>
                  </label>
                  <input
                    type="password"
                    id="GEMMA_API_KEY"
                    name="GEMMA_API_KEY"
                    value={settings.GEMMA_API_KEY}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter your Lightricks Gemma API key"
                  />
                </div>

                <div>
                  <label htmlFor="TRAINING_FOLDER" className="block text-sm font-medium mb-2">
                    Training Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      We will store your training information here. Must be an absolute path. If blank, it will default
                      to the output folder in the project root.
                    </div>
                  </label>
                  <input
                    type="text"
                    id="TRAINING_FOLDER"
                    name="TRAINING_FOLDER"
                    value={settings.TRAINING_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter training folder path"
                  />
                </div>

                <div>
                  <label htmlFor="DATASETS_FOLDER" className="block text-sm font-medium mb-2">
                    Dataset Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      Where we store and find your datasets.{' '}
                      <span className="text-orange-800">
                        Warning: This software may modify datasets so it is recommended you keep a backup somewhere else
                        or have a dedicated folder for this software.
                      </span>
                    </div>
                  </label>
                  <input
                    type="text"
                    id="DATASETS_FOLDER"
                    name="DATASETS_FOLDER"
                    value={settings.DATASETS_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter datasets folder path"
                  />
                </div>

                <div>
                  <label htmlFor="QUANTIZATION_CACHE_DIR" className="block text-sm font-medium mb-2">
                    Quantization Cache Directory
                    <div className="text-gray-500 text-sm ml-1">
                      Where pre-quantized models are stored so the slow quantization step can be skipped
                      on subsequent runs. Must be an absolute path. If blank, defaults to{' '}
                      <code className="text-gray-300">quantized/</code> inside the Training Folder.
                    </div>
                  </label>
                  <input
                    type="text"
                    id="QUANTIZATION_CACHE_DIR"
                    name="QUANTIZATION_CACHE_DIR"
                    value={settings.QUANTIZATION_CACHE_DIR}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Leave blank to use Training Folder/quantized"
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="border-t border-gray-700 pt-6">
            <h2 className="text-base font-semibold mb-4 flex items-center gap-1">
              AI Config Check
              <span className="inline-block ml-1 text-xs text-gray-500 cursor-pointer" onClick={() => openDoc(aiConfigCheckDoc)}>
                <CircleHelp className="inline-block w-4 h-4 cursor-pointer" />
              </span>
            </h2>
            <div className="space-y-4">
              <div>
                <label htmlFor="CHECK_CONFIG_API_BASE_URL" className="block text-sm font-medium mb-2">
                  API Base URL
                  <div className="text-gray-500 text-sm ml-1">
                    OpenAI-compatible endpoint. Use{' '}
                    <code className="text-gray-300">https://api.anthropic.com/v1</code> for Claude,
                    or your local Ollama address (e.g. <code className="text-gray-300">http://192.168.1.x:11434/v1</code>).
                    Leave blank to disable the Check Config button.
                  </div>
                </label>
                <input
                  type="text"
                  id="CHECK_CONFIG_API_BASE_URL"
                  name="CHECK_CONFIG_API_BASE_URL"
                  value={settings.CHECK_CONFIG_API_BASE_URL}
                  onChange={handleChange}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                  placeholder="https://api.anthropic.com/v1"
                />
              </div>

              <div>
                <label htmlFor="CHECK_CONFIG_API_KEY" className="block text-sm font-medium mb-2">
                  API Key
                  <div className="text-gray-500 text-sm ml-1">
                    Your Anthropic or provider API key. For Ollama, leave blank or enter your Ollama
                    API key if your server requires one (also used for web search authentication).
                  </div>
                </label>
                <input
                  type="password"
                  id="CHECK_CONFIG_API_KEY"
                  name="CHECK_CONFIG_API_KEY"
                  value={settings.CHECK_CONFIG_API_KEY}
                  onChange={handleChange}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                  placeholder="sk-ant-..."
                />
              </div>

              <div>
                <label htmlFor="CHECK_CONFIG_MODEL" className="block text-sm font-medium mb-2">
                  Model
                  <div className="text-gray-500 text-sm ml-1">
                    Model to use for config analysis. Defaults to{' '}
                    <code className="text-gray-300">claude-sonnet-5</code>. For visual analysis of
                    sample images, a vision-capable model is required (e.g. Qwen2.5-VL via Ollama).
                  </div>
                </label>
                <input
                  type="text"
                  id="CHECK_CONFIG_MODEL"
                  name="CHECK_CONFIG_MODEL"
                  value={settings.CHECK_CONFIG_MODEL}
                  onChange={handleChange}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                  placeholder="claude-sonnet-5"
                />
              </div>

              <div>
                <label className="flex items-start gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    name="CHECK_CONFIG_ENABLE_WEB_SEARCH"
                    checked={settings.CHECK_CONFIG_ENABLE_WEB_SEARCH === 'true'}
                    onChange={handleChange}
                    className="mt-1 h-4 w-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium">
                    Enable Ollama Web Search
                    <div className="text-gray-500 text-sm font-normal mt-1">
                      When using an Ollama endpoint, perform a web search before the LLM call and
                      inject the results as additional context. Requires the Ollama server to have
                      web search enabled. Has no effect when using Anthropic or other non-Ollama providers.
                    </div>
                  </span>
                </label>
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={status === 'saving'}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status === 'saving' ? 'Saving...' : 'Save Settings'}
          </button>

          {status === 'success' && <p className="text-green-500 text-center">Settings saved successfully!</p>}
          {status === 'error' && <p className="text-red-500 text-center">Error saving settings. Please try again.</p>}
        </form>
      </MainContent>
    </>
  );
}
