import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Stack,
  Typography,
} from "@mui/material";

import { uploadDataset } from "../api/client";
import { PreviewTable } from "../components/PreviewTable";
import { useAppState } from "../context/AppStateContext";

const acceptedExtensions = [".csv", ".xlsx", ".xls"];

export function UploadPage() {
  const navigate = useNavigate();
  const { state, setSession, setTaskConfig, setAnalysis, setReview, setTraining } = useAppState();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string>("");
  const [isUploading, setIsUploading] = useState(false);

  const uploadSummary = useMemo(() => state.session, [state.session]);

  const onFileSelected = (file: File | null) => {
    setError("");
    if (!file) {
      setSelectedFile(null);
      return;
    }

    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!acceptedExtensions.includes(ext)) {
      setError("Unsupported file type. Please upload .csv, .xlsx, or .xls.");
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  };

  const onUpload = async () => {
    if (!selectedFile) {
      setError("Please choose a dataset file first.");
      return;
    }

    try {
      setError("");
      setIsUploading(true);
      const response = await uploadDataset(selectedFile);
      setSession(response);
      setTaskConfig(null);
      setAnalysis(null);
      setReview(null);
      setTraining(null);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Upload failed. Please try again.";
      setError(message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Stack spacing={3}>
      <Card variant="outlined">
        <CardContent>
          <Stack spacing={2}>
            <Typography variant="h5" sx={{ fontWeight: 700 }}>
              1) Upload Dataset
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Upload a CSV or Excel file. The API will create a session and return schema + preview.
            </Typography>

            <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ alignItems: { sm: "center" } }}>
              <Button component="label" variant="contained">
                Select File
                <input
                  hidden
                  type="file"
                  accept={acceptedExtensions.join(",")}
                  onChange={(event) => onFileSelected(event.target.files?.[0] ?? null)}
                />
              </Button>
              <Typography variant="body2">{selectedFile?.name || "No file selected"}</Typography>
            </Stack>

            <Stack direction="row" spacing={2}>
              <Button variant="contained" onClick={onUpload} disabled={isUploading || !selectedFile}>
                {isUploading ? <CircularProgress size={20} /> : "Upload Dataset"}
              </Button>
              <Button
                variant="outlined"
                onClick={() => navigate("/configure")}
                disabled={!state.session}
              >
                Continue to Configure
              </Button>
            </Stack>

            {error && <Alert severity="error">{error}</Alert>}
            {uploadSummary && <Alert severity="success">{uploadSummary.message}</Alert>}
          </Stack>
        </CardContent>
      </Card>

      {uploadSummary && (
        <Card variant="outlined">
          <CardContent>
            <Stack spacing={2}>
              <Typography variant="h6">Dataset Summary</Typography>
              <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: "wrap" }}>
                <Chip label={`Session: ${uploadSummary.session_id.slice(0, 8)}...`} />
                <Chip label={`Rows: ${uploadSummary.shape.rows}`} color="primary" />
                <Chip label={`Columns: ${uploadSummary.shape.columns}`} color="secondary" />
              </Stack>

              <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
                <Box sx={{ width: { xs: "100%", md: "40%" } }}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    Columns
                  </Typography>
                  <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: "wrap" }}>
                    {uploadSummary.columns.map((column) => (
                      <Chip key={column} label={column} size="small" variant="outlined" />
                    ))}
                  </Stack>
                </Box>
                <Box sx={{ width: { xs: "100%", md: "60%" } }}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    Data Types
                  </Typography>
                  <Box sx={{ fontFamily: "monospace", fontSize: 13, whiteSpace: "pre-wrap" }}>
                    {JSON.stringify(uploadSummary.dtypes, null, 2)}
                  </Box>
                </Box>
              </Stack>

              <Typography variant="subtitle2">Preview (first 5 rows)</Typography>
              <PreviewTable rows={uploadSummary.preview} />
            </Stack>
          </CardContent>
        </Card>
      )}
    </Stack>
  );
}
