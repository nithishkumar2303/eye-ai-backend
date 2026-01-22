const express = require("express");
const app = express();

const PORT = process.env.PORT || 8000;

const cors = require("cors");

app.use(cors({
  origin: [
    "http://localhost:5173",
    "https://<project>.vercel.app"
  ],
  credentials: true
}));
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});