import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { getApiBaseUrl } from "../api/client";
import { useAppState } from "../context/AppStateContext";
import type { ClusterProfile } from "../types/api";

const colors = ["#1976d2", "#8e24aa", "#2e7d32", "#ef6c00", "#d32f2f", "#00838f"];

function formatMetricLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function metricData(metrics: Record<string, unknown>, keys: string[]) {
  return keys
    .filter((key) => typeof metrics[key] === "number")
    .map((key) => ({ name: formatMetricLabel(key), value: Number(metrics[key]) }));
}

export function ResultsPage() {
  const navigate = useNavigate();
  const { state } = useAppState();
  const training = state.training;
  const taskConfig = state.taskConfig;
  const session = state.session;

  const apiBase = getApiBaseUrl();

  const evaluation = useMemo(() => training?.evaluation || {}, [training]);

  if (!training || !taskConfig || !session) {
    return (
      <Card variant="outlined">
        <CardContent>
          <Stack spacing={2}>
            <Alert severity="warning">No training results found. Complete the Review step first.</Alert>
            <Button variant="contained" onClick={() => navigate("/review")}>Go to Review</Button>
          </Stack>
        </CardContent>
      </Card>
    );
  }

  const downloadUrl = `${apiBase}/download/${session.session_id}`;

  const classificationKeys = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "f1_macro",
  ];
  const regressionKeys = ["mae", "mse", "rmse", "r2"];
  const clusteringKeys = ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"];

  const selectedKeys =
    taskConfig.task_type === "classification"
      ? classificationKeys
      : taskConfig.task_type === "regression"
        ? regressionKeys
        : clusteringKeys;

  const chartData = metricData(evaluation, selectedKeys);
  const featureData = training.feature_importance || [];

  const confusionMatrix = Array.isArray(evaluation.confusion_matrix)
    ? (evaluation.confusion_matrix as number[][])
    : null;
  const clusterLabelsMap = (evaluation.cluster_labels_map || {}) as Record<string, string>;
  const clusterProfiles = (evaluation.cluster_profiles || {}) as Record<string, ClusterProfile>;
  const clusterSizes = (evaluation.cluster_sizes || {}) as Record<string, number>;

  const matrixMax = confusionMatrix
    ? Math.max(...confusionMatrix.flat().map((value) => Number(value))) || 1
    : 1;

  return (
    <Stack spacing={3}>
      <Card variant="outlined">
        <CardContent>
          <Stack spacing={2}>
            <Typography variant="h5" sx={{ fontWeight: 700 }}>
              4) Results & Export
            </Typography>

            <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: "wrap" }}>
              <Chip label={`Best Model: ${training.best_model.name}`} color="primary" />
              <Chip label={`Score: ${training.best_model.score}`} color="secondary" />
              <Chip label={`Task: ${taskConfig.task_type}`} variant="outlined" />
            </Stack>

            <Stack direction="row" spacing={2}>
              <Button variant="outlined" onClick={() => navigate("/review")}>Back</Button>
              <Button
                variant="contained"
                component="a"
                href={downloadUrl}
                target="_blank"
                rel="noreferrer"
              >
                Download Model (.joblib)
              </Button>
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Preprocessing Report
          </Typography>
          <Stack spacing={1}>
            {training.preprocessing_report.map((line, index) => (
              <Alert key={index} severity="info" icon={false}>
                {line}
              </Alert>
            ))}
          </Stack>
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Model Comparison
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Algorithm</TableCell>
                  <TableCell>CV Score</TableCell>
                  <TableCell>Best Params</TableCell>
                  <TableCell>Error</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(training.model_comparison).map(([name, result]) => (
                  <TableRow key={name}>
                    <TableCell>{name}</TableCell>
                    <TableCell>{result.cv_score ?? "N/A"}</TableCell>
                    <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>
                      {JSON.stringify(result.best_params || {})}
                    </TableCell>
                    <TableCell>{result.error || "-"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Evaluation Metrics
          </Typography>
          {chartData.length > 0 ? (
            <Box sx={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <BarChart data={chartData} margin={{ left: 10, right: 10, top: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" interval={0} angle={-15} textAnchor="end" height={70} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value">
                    {chartData.map((entry, index) => (
                      <Cell key={entry.name} fill={colors[index % colors.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Box>
          ) : (
            <Alert severity="info">No numeric evaluation metrics were returned.</Alert>
          )}
        </CardContent>
      </Card>

      {taskConfig.task_type === "classification" && confusionMatrix && (
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Confusion Matrix
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableBody>
                  {confusionMatrix.map((row, rowIndex) => (
                    <TableRow key={rowIndex}>
                      {row.map((value, colIndex) => {
                        const intensity = Number(value) / matrixMax;
                        return (
                          <TableCell
                            key={`${rowIndex}-${colIndex}`}
                            sx={{
                              textAlign: "center",
                              backgroundColor: `rgba(25, 118, 210, ${Math.max(0.15, intensity)})`,
                              color: intensity > 0.45 ? "white" : "black",
                              fontWeight: 700,
                            }}
                          >
                            {value}
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {taskConfig.task_type === "clustering" && Object.keys(clusterLabelsMap).length > 0 && (
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Cluster Definitions
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Cluster ID</TableCell>
                    <TableCell>Auto Label</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Profile Signals</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(clusterLabelsMap).map(([clusterId, label]) => (
                    <TableRow key={clusterId}>
                      <TableCell>{clusterId}</TableCell>
                      <TableCell>{label}</TableCell>
                      <TableCell>{clusterSizes[clusterId] ?? "-"}</TableCell>
                      <TableCell>
                        {(clusterProfiles[clusterId]?.top_numeric_signals || []).join(" | ") || "-"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Feature Importance (Top 20)
          </Typography>
          {featureData.length > 0 ? (
            <Box sx={{ width: "100%", height: 360 }}>
              <ResponsiveContainer>
                <BarChart data={featureData} layout="vertical" margin={{ left: 50, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="feature" width={160} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#6a1b9a" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          ) : (
            <Alert severity="info">
              Selected best model does not expose feature importance.
            </Alert>
          )}
        </CardContent>
      </Card>
    </Stack>
  );
}
