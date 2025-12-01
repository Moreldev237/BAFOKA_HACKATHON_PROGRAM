import express from "express";
import dotenv from "dotenv";
import transcribeRoutes from "./routes/transcribe.js";

dotenv.config();
const app = express();

app.use(express.json());

// Routes
app.use("/api/transcribe", transcribeRoutes);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
