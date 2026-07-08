# Wan2.2 TI2V 5B Modly Extension

Modly model extension for image-to-video generation with **Wan2.2 TI2V 5B**.

This repository contains the Modly wrapper, setup script, manifest, and a vendored Wan runtime snapshot. Model weights are not bundled and are downloaded from Modly's Models UI after setup.

## GitHub About

- Description: `Wan2.2 TI2V 5B image-to-video extension for Modly`
- Website: `https://github.com/Wan-Video/Wan2.2`
- Topics: `modly-extension`, `image-to-video`, `video-generation`, `wan22`, `wan-ai`, `ti2v`, `cuda`, `pytorch`, `flash-attn`, `mit`

## Modly contract

- Extension type: `model`
- Bucket: `model-managed-setup`
- Setup entrypoint: `setup.py`
- Generator entrypoint: `generator.py` / `Wan22TI2VGenerator`
- Model weights: `Wan-AI/Wan2.2-TI2V-5B`
- Runtime policy: `setup.py` prepares the Python environment only; it never clones Wan and never downloads model weights.
- Output: workflow-compatible `video` artifact (`.mp4`).

## Workflow video preview node

The manifest declares a reusable workflow utility node:

- Capability: `modly.workflow.preview.video`
- Component: `video-preview`
- Singleton: `true`

Modly uses this declaration to show one **Preview Video** node when this extension is installed. Future image-to-video or video-to-video extensions can declare the same capability and Modly will avoid duplicate player nodes.

## Setup

Install the extension through Modly's GitHub extension installer, then run setup/repair from the Models UI.

The setup script will:

1. create `venv/`;
2. install the pinned Python runtime requirements;
3. prepare the vendored Wan runtime for import;
4. write observable setup status/log output for Modly.

It will not clone upstream Wan and will not download model weights.

## Usage

1. Open Modly.
2. Install/repair the extension until setup is ready.
3. Download the Wan weights from the Models UI.
4. In Workflows, connect an image to **Image to Video**.
5. Connect the output to **Preview Video** to play the generated MP4.

## Performance and thermals

Video generation is a sustained GPU workload. The manifest defaults to safer first-run settings (`81` frames, `20` steps). Increase frames or steps only after validating GPU and system thermals on your machine.

A 5-second generation (`121` frames, `20` steps) can take around tens of minutes depending on hardware and cooling.

## License

The Modly wrapper code in this repository is licensed under the MIT License. See `LICENSE`.

Wan2.2 runtime code and model weights are third-party components with their own licenses and terms. See `THIRD_PARTY_NOTICES.md` and `vendor/wan-runtime/LICENSE.txt`.
