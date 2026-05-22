import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  CircularProgress,
  Divider,
  FormControl,
  FormControlLabel,
  FormGroup,
  InputLabel,
  List,
  ListItem,
  ListItemText,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";

import { trainModel } from "../api/client";
import { useAppState } from "../context/AppStateContext";
import type { AnalyzeQuestion, ReviewConfig, TrainRequest } from "../types/api";

const PREPROCESS_IDS = new Set([
  "missing_values",
  "duplicates",
  "outliers",
  "categorical_encoding",
  "high_cardinality",
  "scaling",
  "class_imbalance",
  "high_dimensionality",
]);

function normalizeChoiceKey(issueId: string, optionKey: string): string {
  if (issueId === "high_cardinality" && optionKey === "strategy") return "high_cardinality_strategy";
  if (issueId === "high_dimensionality" && optionKey === "n_components") return "pca_n_components";
  return optionKey;
}

function toPrettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function toErrorMessage(error: unknown): string {
  if (typeof error === "object" && error && "response" in error) {
    const maybeResponse = error as { response?: { data?: { detail?: string } } };
    return maybeResponse.response?.data?.detail || "Request failed.";
  }
  if (error instanceof Error) return error.message;
  return "Request failed.";
}

function detailToString(detail: unknown): string {
  if (detail === null || detail === undefined) return "";
  if (typeof detail === "string") return detail;
  return JSON.stringify(detail, null, 2);
}

function buildInitialReviewConfig(
  existing: ReviewConfig | null,
  defaults: Record<string, unknown>,
  defaultParamSpace: Record<string, unknown>,
): ReviewConfig {
  if (existing) return existing;
  return {
    choices: { ...defaults },
    use_default_algorithms: true,
    selected_algorithms: [],
    use_grid_search: true,
    param_mode: "default",
    custom_params_text: toPrettyJson(defaultParamSpace),
  };
}

export function ReviewPage() {
  const navigate = useNavigate();
  const { state, setReview, setTraining } = useAppState();

  const [reviewConfig, setReviewConfig] = useState<ReviewConfig | null>(null);
  const [error, setError] = useState("");
  const [isTraining, setIsTraining] = useState(false);

  const analysis = state.analysis;
  const taskConfig = state.taskConfig;
  const session = state.session;
  const isClustering = taskConfig?.task_type === "clustering";

  useEffect(() => {
    if (!analysis || !taskConfig || !session) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setReviewConfig(
      buildInitialReviewConfig(
        state.review,
        analysis.default_choices,
        analysis.default_param_space,
      ),
    );
  }, [analysis, taskConfig, session, state.review]);

  const preprocessQuestions = useMemo(
    () => (analysis?.questions_for_user || []).filter((q) => PREPROCESS_IDS.has(q.id)),
    [analysis],
  );

  const updateChoice = (key: string, value: unknown) => {
    setReviewConfig((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        choices: {
          ...prev.choices,
          [key]: value,
        },
      };
    });
  };

  const setAlgorithmSelectionMode = (useDefaults: boolean) => {
    setReviewConfig((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        use_default_algorithms: useDefaults,
      };
    });
  };

  const toggleSelectedAlgorithm = (algorithm: string) => {
    setReviewConfig((prev) => {
      if (!prev) return prev;
      const exists = prev.selected_algorithms.includes(algorithm);
      const selected = exists
        ? prev.selected_algorithms.filter((item) => item !== algorithm)
        : [...prev.selected_algorithms, algorithm];

      return { ...prev, selected_algorithms: selected };
    });
  };

  const canTrain = useMemo(() => {
    if (!reviewConfig || !analysis) return false;
    if (reviewConfig.use_default_algorithms) return true;
    return reviewConfig.selected_algorithms.length >= 2;
  }, [analysis, reviewConfig]);

  const runTraining = async () => {
    if (!analysis || !taskConfig || !session || !reviewConfig) return;

    if (!reviewConfig.use_default_algorithms && reviewConfig.selected_algorithms.length < 2) {
      setError("Please select at least two algorithms when custom mode is enabled.");
      return;
    }

    let customParams: TrainRequest["custom_params"];
    if (reviewConfig.param_mode === "custom" && !isClustering) {
      try {
        customParams = JSON.parse(reviewConfig.custom_params_text);
      } catch {
        setError("Custom parameter JSON is invalid. Please fix it and try again.");
        return;
      }
    }

    try {
      setError("");
      setIsTraining(true);

      const payload: TrainRequest = {
        session_id: session.session_id,
        task_type: taskConfig.task_type,
        target_col: taskConfig.target_col,
        choices: reviewConfig.choices,
        selected_algorithms: reviewConfig.use_default_algorithms
          ? undefined
          : reviewConfig.selected_algorithms,
        use_default_algorithms: reviewConfig.use_default_algorithms,
        use_grid_search: reviewConfig.use_grid_search,
        custom_params: isClustering ? undefined : customParams,
      };

      const result = await trainModel(payload);
      setReview(reviewConfig);
      setTraining(result);
      navigate("/results");
    } catch (err: unknown) {
      setError(toErrorMessage(err));
    } finally {
      setIsTraining(false);
    }
  };

  if (!analysis || !taskConfig || !session || !reviewConfig) {
    return (
      <Card variant="outlined">
        <CardContent>
          <Stack spacing={2}>
            <Alert severity="warning">Missing analysis context. Run Configure step first.</Alert>
            <Button variant="contained" onClick={() => navigate("/configure")}>Go to Configure</Button>
          </Stack>
        </CardContent>
      </Card>
    );
  }

  const renderIssueCard = (question: AnalyzeQuestion) => {
    const enabled = Boolean(reviewConfig.choices[question.id] ?? true);
    const options = question.options || {};

    return (
      <Card key={question.id} variant="outlined">
        <CardContent>
          <Stack spacing={1.5}>
            <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center" }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                {question.question}
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={enabled}
                    onChange={(event) => updateChoice(question.id, event.target.checked)}
                  />
                }
                label={enabled ? "Apply" : "Skip"}
              />
            </Stack>

            {Boolean(question.detail) && (
              <Box
                sx={{
                  p: 1.25,
                  borderRadius: 1,
                  bgcolor: "action.hover",
                  fontFamily: "monospace",
                  fontSize: 12,
                  whiteSpace: "pre-wrap",
                }}
              >
                {detailToString(question.detail)}
              </Box>
            )}

            {enabled &&
              Object.entries(options).map(([optionKey, optionValue]) => {
                const key = normalizeChoiceKey(question.id, optionKey);
                const currentValue = reviewConfig.choices[key] ?? question.defaults?.[optionKey] ?? "";

                if (Array.isArray(optionValue)) {
                  return (
                    <FormControl fullWidth size="small" key={`${question.id}-${optionKey}`}>
                      <InputLabel>{optionKey}</InputLabel>
                      <Select
                        value={String(currentValue)}
                        label={optionKey}
                        onChange={(event) => updateChoice(key, event.target.value)}
                      >
                        {optionValue.map((value) => (
                          <MenuItem key={String(value)} value={String(value)}>
                            {String(value)}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  );
                }

                return (
                  <TextField
                    key={`${question.id}-${optionKey}`}
                    label={optionKey}
                    value={String(currentValue)}
                    onChange={(event) => updateChoice(key, event.target.value)}
                    fullWidth
                    size="small"
                  />
                );
              })}
          </Stack>
        </CardContent>
      </Card>
    );
  };

  return (
    <Stack spacing={3}>
      <Card variant="outlined">
        <CardContent>
          <Stack spacing={2}>
            <Typography variant="h5" sx={{ fontWeight: 700 }}>
              3) Review Recommendations & Training Options
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Confirm preprocessing actions and choose model search preferences.
            </Typography>
          </Stack>
        </CardContent>
      </Card>

      <Stack spacing={2}>{preprocessQuestions.map(renderIssueCard)}</Stack>

      <Card variant="outlined">
        <CardContent>
          <Stack spacing={2}>
            <Typography variant="h6">Algorithm Selection</Typography>
            <RadioGroup
              value={reviewConfig.use_default_algorithms ? "default" : "custom"}
              onChange={(event) => setAlgorithmSelectionMode(event.target.value === "default")}
            >
              <FormControlLabel value="default" control={<Radio />} label="Use default algorithm set" />
              <FormControlLabel value="custom" control={<Radio />} label="Choose custom algorithms" />
            </RadioGroup>

            {!reviewConfig.use_default_algorithms && (
              <FormGroup>
                {analysis.available_algorithms.map((algorithm) => (
                  <FormControlLabel
                    key={algorithm}
                    control={
                      <Checkbox
                        checked={reviewConfig.selected_algorithms.includes(algorithm)}
                        onChange={() => toggleSelectedAlgorithm(algorithm)}
                      />
                    }
                    label={algorithm}
                  />
                ))}
              </FormGroup>
            )}

            {!reviewConfig.use_default_algorithms && reviewConfig.selected_algorithms.length < 2 && (
              <Alert severity="warning">Select at least two algorithms in custom mode.</Alert>
            )}

            <Divider />

            <Typography variant="h6">Search Space and Grid Search</Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={reviewConfig.use_grid_search}
                  onChange={(event) =>
                    setReviewConfig((prev) =>
                      prev ? { ...prev, use_grid_search: event.target.checked } : prev,
                    )
                  }
                />
              }
              label="Use Grid Search"
            />

            <RadioGroup
              value={reviewConfig.param_mode}
              onChange={(event) =>
                setReviewConfig((prev) =>
                  prev
                    ? {
                        ...prev,
                        param_mode: event.target.value as ReviewConfig["param_mode"],
                      }
                    : prev,
                )
              }
            >
              <FormControlLabel value="default" control={<Radio />} label="Use default parameter space" />
              <FormControlLabel
                value="custom"
                control={<Radio />}
                label={isClustering ? "Provide custom parameter JSON (disabled for clustering)" : "Provide custom parameter JSON"}
                disabled={isClustering}
              />
            </RadioGroup>

            {isClustering && (
              <Alert severity="info">
                For clustering, KMeans cluster count is selected automatically by silhouette score; custom k values are ignored.
              </Alert>
            )}

            {reviewConfig.param_mode === "custom" && !isClustering && (
              <TextField
                label="custom_params JSON"
                value={reviewConfig.custom_params_text}
                onChange={(event) =>
                  setReviewConfig((prev) =>
                    prev ? { ...prev, custom_params_text: event.target.value } : prev,
                  )
                }
                multiline
                minRows={10}
                fullWidth
              />
            )}

            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Current Default Search Space
              </Typography>
              <Box
                sx={{
                  p: 1.25,
                  borderRadius: 1,
                  bgcolor: "action.hover",
                  fontFamily: "monospace",
                  fontSize: 12,
                  whiteSpace: "pre-wrap",
                }}
              >
                {toPrettyJson(analysis.default_param_space)}
              </Box>
            </Box>

            <Divider />
            <Typography variant="h6">Train/Test Split</Typography>
            <TextField
              label="test_size"
              type="number"
              slotProps={{ htmlInput: { min: 0.1, max: 0.5, step: 0.05 } }}
              value={String(reviewConfig.choices.test_size ?? 0.2)}
              onChange={(event) => updateChoice("test_size", Number(event.target.value))}
              disabled={taskConfig.task_type === "clustering"}
              sx={{ maxWidth: 220 }}
            />

            <Typography variant="h6">Detected Backend Questions</Typography>
            <List dense>
              {analysis.questions_for_user.map((q) => (
                <ListItem key={q.id}>
                  <ListItemText primary={q.question} secondary={`id: ${q.id}`} />
                </ListItem>
              ))}
            </List>

            <Stack direction="row" spacing={2}>
              <Button variant="outlined" onClick={() => navigate("/configure")}>Back</Button>
              <Button variant="contained" onClick={runTraining} disabled={!canTrain || isTraining}>
                {isTraining ? <CircularProgress size={20} /> : "Train Models"}
              </Button>
            </Stack>

            {error && <Alert severity="error">{error}</Alert>}
          </Stack>
        </CardContent>
      </Card>
    </Stack>
  );
}
