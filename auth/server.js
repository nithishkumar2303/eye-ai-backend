const express = require("express");
const app = express();

const PORT = process.env.PORT || 8000;

// middleware + routes here
// app.use(...)

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});