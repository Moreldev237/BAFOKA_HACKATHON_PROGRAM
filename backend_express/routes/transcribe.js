import express from "express";
import multer from "multer";
import { transcribeAudio } from "../controllers/transcribeController.js";

const router = express.Router();

// Config multer pour upload temporaire
const upload = multer({ dest: "uploads/" });

router.post("/", upload.single("audio"), transcribeAudio);

export default router;
