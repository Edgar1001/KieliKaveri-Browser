# LanguageTutor

LanguageTutor is an Expo/React Native prototype for practising spoken Finnish. Record a short turn, receive a transcript and one useful correction, then hear the tutor continue the conversation in Finnish.

## Architecture

The mobile app sends each recording to a FastAPI service, which performs three steps:

1. Finnish speech-to-text with faster-whisper and Whisper large-v3.
2. Correction and conversation generation with OpenAI GPT-4.1.
3. Finnish text-to-speech with Android `expo-speech` or the local Piper neural voice in the Linux browser.

This first prototype corrects grammar, vocabulary, and natural phrasing from the transcript. Reliable pronunciation scoring needs a later phoneme/audio analysis stage and is deliberately not inferred from transcript text.

Only the transcript and recent conversation context are sent to OpenAI. Audio transcription and browser speech synthesis remain local. The full `large-v3` model is used for Finnish accuracy; set `WHISPER_MODEL=turbo` to trade some accuracy for speed.

The app starts at CEFR `B2` and `Puhekieli`. Select a level from A2 through C2 to control the tutor's vocabulary, sentence structure, and reply length. Select `Puhekieli` for neutral conversational Finnish or `Kirjakieli` for standard written Finnish. Corrections follow the selected register, so established colloquial forms are accepted in `Puhekieli` mode. Each tutor reply includes a small English translation. For lower API cost and latency, set `OPENAI_MODEL=gpt-4.1-mini`; the default `gpt-4.1` prioritizes correction accuracy.

## Run locally

Requirements: Node.js 20+, Python 3.11+, an OpenAI API key, an NVIDIA driver with CUDA 12 support, and Expo Go 57 on an Android phone or emulator.

```bash
cp .env.example .env
npm install
npm run local:setup
```

Open `.env` and set the API key created at [platform.openai.com/api-keys](https://platform.openai.com/api-keys):

```dotenv
OPENAI_API_KEY=your-key-here
```

Keep this key only on the FastAPI server; do not place it in an `EXPO_PUBLIC_` variable or bundle it into the app. Start the local API and Expo:

```bash
npm run dev
```

The supplied `.env.example` uses `http://10.0.2.2:8787`, Android Emulator's alias for the development computer. For a physical phone, replace it with the computer's LAN address, such as `http://192.168.1.100:8787`, and restart Expo.

Open the project in Expo Go, tap the microphone, speak Finnish for a few seconds, and tap stop. Recordings are capped at 60 seconds. The first response is slower while Whisper loads for the first time.

To run the services separately for troubleshooting:

```bash
npm run api
npm start
```

For CPU-only transcription, set `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8` in `.env`.

## Checks

```bash
npm run typecheck
npm test
```

## Build an Android APK

Sign in to Expo, configure the project when prompted, and run the internal-distribution profile:

```bash
npx eas-cli@latest login
npx eas-cli@latest build --platform android --profile preview
```

EAS provides a download link for the installable `.apk`. A phone cannot use the emulator-only `10.0.2.2` address; set `EXPO_PUBLIC_API_URL` to a FastAPI host reachable by the phone before building. Configure `OPENAI_API_KEY` only on that host, never in the APK.