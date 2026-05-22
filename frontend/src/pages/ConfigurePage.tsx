import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Button,
  Card,
  CardContent,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";

import { analyzeDataset } from "../api/client";
import { useAppState } from "../context/AppStateContext";
import type { TaskType } from "../types/api";

export function ConfigurePage() {
  const navigate = useNavigate();
  const { state, setTaskConfig, setAnalysis, setReview, setTraining } = useAppState();

  const [taskType, setTaskType] = useState<TaskType>(state.taskConfig?.task_type || "classification");
  const [targetCol, setTargetCol] = useState(state.taskConfig?.target_col || "");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");

  const isSupervised = taskType === "classification" || taskType === "regression";

  const analyze = async () => {
    if (!state.session) {
      setError("Please upload a dataset before analysis.");
      return;
    }

    if (isSupervised && !targetCol) {
      setError("Please select a target column for supervised learning.");
      return;
    }

    try {
      setIsAnalyzing(true);
      setError("");

      const config = {
        task_type: taskType,
        ...(isSupervised ? { target_col: targetCol } : {}),
      };

      const response = await analyzeDataset(state.session.session_id, config);

      setTaskConfig(config);
      setAnalysis(response);
      setReview(null);
      setTraining(null);
      navigate("/review");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Analyze request failed.";
      setError(message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <Stack spacing={3}>
      <Card variant="outlined">
        <CardContent>
          <Stack spacing={2.5}>
            <Typography variant="h5" sx={{ fontWeight: 700 }}>
              2) Configure Task
            </Typography>

            {!state.session && (
              <Alert severity="warning">
                Upload is required first. Return to the upload step before continuing.
              </Alert>
            )}

            <Stack spacing={1}>
              <Typography variant="subtitle1">ML Problem Type</Typography>
              <ToggleButtonGroup
                exclusive
                value={taskType}
                onChange={(_, value: TaskType | null) => {
                  if (value) setTaskType(value);
                }}
              >
                <ToggleButton value="classification">Classification</ToggleButton>
                <ToggleButton value="regression">Regression</ToggleButton>
                <ToggleButton value="clustering">Clustering</ToggleButton>
              </ToggleButtonGroup>
            </Stack>

            {isSupervised && (
              <FormControl fullWidth>
                <InputLabel id="target-column-label">Target Column</InputLabel>
                <Select
                  labelId="target-column-label"
                  value={targetCol}
                  label="Target Column"
                  onChange={(e) => setTargetCol(e.target.value)}
                >
                  {(state.session?.columns || []).map((column) => (
                    <MenuItem key={column} value={column}>
                      {column}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}

            <Stack direction="row" spacing={2}>
              <Button variant="outlined" onClick={() => navigate("/")}>
                Back to Upload
              </Button>
              <Button
                variant="contained"
                onClick={analyze}
                disabled={!state.session || isAnalyzing || (isSupervised && !targetCol)}
              >
                {isAnalyzing ? <CircularProgress size={20} /> : "Analyze Dataset"}
              </Button>
            </Stack>

            {error && <Alert severity="error">{error}</Alert>}
            {state.analysis && <Alert severity="success">{state.analysis.message}</Alert>}
          </Stack>
        </CardContent>
      </Card>
    </Stack>
  );
}
