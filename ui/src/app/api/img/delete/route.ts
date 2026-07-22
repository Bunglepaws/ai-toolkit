import { NextResponse } from 'next/server';
import fs from 'fs';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

const fileExists = (p: string) => fs.promises.access(p).then(() => true).catch(() => false);

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath } = body;
    let datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();

    // make sure the dataset path is in the image path
    if (!imgPath.startsWith(datasetsPath) && !imgPath.startsWith(trainingPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // make sure it is a supported media type (keep in sync with ui/src/utils/basic.ts)
    if (!/\.(jpg|jpeg|png|bmp|gif|webp|svg|mp4|avi|mov|mkv|wmv|m4v|flv|mp3|wav|flac|ogg|m4a)$/i.test(imgPath.toLowerCase())) {
      return NextResponse.json({ error: 'Not an image' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    if (!(await fileExists(imgPath))) {
      return NextResponse.json({ success: true });
    }

    // delete it and return success
    await fs.promises.unlink(imgPath);

    // delete any companion caption files, regardless of which extension is active
    const filenameNoExt = imgPath.replace(/\.[^/.]+$/, '');
    for (const ext of ['txt', 'json', 'caption']) {
      const captionPath = `${filenameNoExt}.${ext}`;
      if (await fileExists(captionPath)) {
        await fs.promises.unlink(captionPath);
      }
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
