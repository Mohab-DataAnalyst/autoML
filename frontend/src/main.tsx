import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { CssBaseline, ThemeProvider, createTheme } from "@mui/material";

import App from "./App";
import { AppStateProvider } from "./context/AppStateContext";
import "./index.css";

const theme = createTheme({
  palette: {
    primary: {
      main: "#0f4c81",
    },
    secondary: {
      main: "#7d2e68",
    },
    background: {
      default: "#f5f7fb",
    },
  },
  shape: {
    borderRadius: 10,
  },
  typography: {
    fontFamily: "'Source Sans 3', 'Segoe UI', sans-serif",
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <AppStateProvider>
          <App />
        </AppStateProvider>
      </BrowserRouter>
    </ThemeProvider>
  </StrictMode>,
);
