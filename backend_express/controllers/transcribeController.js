import fs from "fs";
import path from "path";
import axios from "axios";
import ffmpeg from "fluent-ffmpeg";
import ffmpegInstaller from "@ffmpeg-installer/ffmpeg";
import { AssemblyAI } from "assemblyai";
import { detectIntent } from "../services/intentService.js";

ffmpeg.setFfmpegPath(ffmpegInstaller.path);

// Convert ANY audio → WAV 48kHz PCM
async function convertToWav(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .audioCodec("pcm_s16le")
      .audioFrequency(48000)
      .audioChannels(2)
      .toFormat("wav")
      .on("end", () => resolve(outputPath))
      .on("error", reject)
      .save(outputPath);
  });
}

// Download audio from URL → file
async function downloadAudioFromUrl(url, savePath) {
  const response = await axios.get(url, {
    responseType: "arraybuffer"
  });

  fs.writeFileSync(savePath, Buffer.from(response.data));
  return savePath;
}

export const transcribeAudio = async (req, res) => {
  const client = new AssemblyAI({
    apiKey: process.env.ASSEMBLYAI_API_KEY,
  });

  let inputPath = null;
  let wavPath = null;

  try {
    // ------------------------------------------------------
    // 1️⃣ GET AUDIO : upload file OR URL from Botpress
    // ------------------------------------------------------
    if (req.file) {
      // FILE UPLOAD
      inputPath = req.file.path;
    } else if (req.body.audio_url) {
      // URL FROM BOTPRESS
      const audioUrl = req.body.audio_url;
      const fileName = `downloaded_${Date.now()}.audio`;
      inputPath = path.join("uploads", fileName);

      await downloadAudioFromUrl(audioUrl, inputPath);
    } else {
      return res.status(400).json({
        error: "No audio file or URL provided"
      });
    }

    // Output WAV path
    wavPath = path.join("uploads", `${Date.now()}.wav`);

    // ------------------------------------------------------
    // 2️⃣ Convert to WAV
    // ------------------------------------------------------
    await convertToWav(inputPath, wavPath);

    // ------------------------------------------------------
    // 3️⃣ Transcribe (with language auto detection)
    // ------------------------------------------------------
    const transcript = await client.transcripts.transcribe({
      audio: fs.createReadStream(wavPath),
      speech_model: "universal",
      language_detection: true,
     // language_code: "fr",
      format_text: true,
      punctuate: true,
    });

    const text = transcript.text?.trim() || "";

    // Clean temp files
    if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath);
    if (fs.existsSync(wavPath)) fs.unlinkSync(wavPath);

    if (!text) {
      return res.json({ text: "", language: null, intent: "autre" });
    }

    // ------------------------------------------------------
    // 4️⃣ Language detected by AssemblyAI
    // ------------------------------------------------------
    let detectedLang = transcript.language_code || "fr";

    if (detectedLang.startsWith("en")) detectedLang = "en";
    if (detectedLang.startsWith("fr")) detectedLang = "fr";

    console.log("Detected language:", detectedLang);

    // ------------------------------------------------------
    // 5️⃣ Intent detection via OpenRouter
    // ------------------------------------------------------
    const intent = await detectIntent(text, detectedLang);

    return res.json({
      text,
      language: detectedLang,
      intent
    });

  } catch (err) {
    console.error("❌ ERROR transcribeAudio :", err);

    if (inputPath && fs.existsSync(inputPath)) fs.unlinkSync(inputPath);
    if (wavPath && fs.existsSync(wavPath)) fs.unlinkSync(wavPath);

    return res.status(500).json({
      error: "Transcription failed",
      detail: err.message,
    });
  }
};
