import ffmpeg from "fluent-ffmpeg";
import ffmpegInstaller from "@ffmpeg-installer/ffmpeg";
import path from "path";

ffmpeg.setFfmpegPath(ffmpegInstaller.path);

function convertToHighQualityWav(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .audioCodec("pcm_s16le")
      .audioFrequency(48000)
      .audioChannels(2)
      .toFormat("wav")
      .on("end", () => {
        console.log("Conversion finished:", outputPath);
        resolve(outputPath);
      })
      .on("error", (err) => {
        console.error("Conversion error:", err);
        reject(err);
      })
      .save(outputPath);
  });
}

// Chemins absolus
const inputFile = path.resolve("./testaudio/1.ogg"); 
const outputFile = path.resolve("./testaudio/1.wav");

convertToHighQualityWav(inputFile, outputFile);
