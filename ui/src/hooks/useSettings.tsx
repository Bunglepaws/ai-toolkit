'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface Settings {
  HF_TOKEN: string;
  GEMMA_API_KEY: string;
  TRAINING_FOLDER: string;
  DATASETS_FOLDER: string;
  QUANTIZATION_CACHE_DIR: string;
  CHECK_CONFIG_API_BASE_URL: string;
  CHECK_CONFIG_API_KEY: string;
  CHECK_CONFIG_MODEL: string;
  CHECK_CONFIG_ENABLE_WEB_SEARCH: string;
  VERSION: string;
}

export default function useSettings() {
  const [settings, setSettings] = useState({
    HF_TOKEN: '',
    GEMMA_API_KEY: '',
    TRAINING_FOLDER: '',
    DATASETS_FOLDER: '',
    QUANTIZATION_CACHE_DIR: '',
    CHECK_CONFIG_API_BASE_URL: '',
    CHECK_CONFIG_API_KEY: '',
    CHECK_CONFIG_MODEL: '',
    CHECK_CONFIG_ENABLE_WEB_SEARCH: 'false',
    VERSION: '',
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    apiClient
      .get('/api/settings')
      .then(res => res.data)
      .then(data => {
        console.log('Settings:', data);
        setSettings({
          HF_TOKEN: data.HF_TOKEN || '',
          GEMMA_API_KEY: data.GEMMA_API_KEY || '',
          TRAINING_FOLDER: data.TRAINING_FOLDER || '',
          DATASETS_FOLDER: data.DATASETS_FOLDER || '',
          QUANTIZATION_CACHE_DIR: data.QUANTIZATION_CACHE_DIR || '',
          CHECK_CONFIG_API_BASE_URL: data.CHECK_CONFIG_API_BASE_URL || '',
          CHECK_CONFIG_API_KEY: data.CHECK_CONFIG_API_KEY || '',
          CHECK_CONFIG_MODEL: data.CHECK_CONFIG_MODEL || '',
          CHECK_CONFIG_ENABLE_WEB_SEARCH: data.CHECK_CONFIG_ENABLE_WEB_SEARCH || 'false',
          VERSION: data.VERSION || '',
        });
        setIsLoaded(true);
      })
      .catch(error => console.error('Error fetching settings:', error));
  }, []);

  return { settings, setSettings, isSettingsLoaded };
}
