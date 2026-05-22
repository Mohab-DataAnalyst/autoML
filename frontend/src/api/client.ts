import axios from "axios";

import type {
  AnalyzeResponse,
  TaskConfig,
  TrainRequest,
  TrainResponse,
  UploadResponse,
} from "../types/api";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export const apiClient = axios.create({
  baseURL,
  timeout: 180000,
});

export function getApiBaseUrl(): string {
  return baseURL;
}

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await apiClient.post<UploadResponse>("/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return data;
}

export async function analyzeDataset(
  sessionId: string,
  taskConfig: TaskConfig,
): Promise<AnalyzeResponse> {
  const { data } = await apiClient.post<AnalyzeResponse>("/analyze", {
    session_id: sessionId,
    task_type: taskConfig.task_type,
    target_col: taskConfig.target_col,
  });

  return data;
}

export async function trainModel(payload: TrainRequest): Promise<TrainResponse> {
  const { data } = await apiClient.post<TrainResponse>("/train", payload);
  return data;
}
