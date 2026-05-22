export type TaskType = "classification" | "regression" | "clustering";

export interface UploadResponse {
  session_id: string;
  filename: string;
  shape: { rows: number; columns: number };
  columns: string[];
  dtypes: Record<string, string>;
  preview: Record<string, unknown>[];
  message: string;
}

export interface AnalyzeQuestion {
  id: string;
  question: string;
  detail?: unknown;
  options?: Record<string, unknown>;
  defaults?: Record<string, unknown>;
  actionable?: boolean;
}

export interface AnalyzeResponse {
  session_id: string;
  task_type: TaskType;
  target_col?: string | null;
  questions_for_user: AnalyzeQuestion[];
  available_algorithms: string[];
  default_choices: Record<string, unknown>;
  default_param_space: Record<string, unknown>;
  message: string;
}

export interface TrainRequest {
  session_id: string;
  task_type: TaskType;
  target_col?: string;
  choices: Record<string, unknown>;
  selected_algorithms?: string[];
  use_default_algorithms: boolean;
  use_grid_search: boolean;
  custom_params?: Record<string, Record<string, unknown[]>>;
}

export interface TrainResponse {
  session_id: string;
  preprocessing_report: string[];
  selected_algorithms: string[];
  model_comparison: Record<
    string,
    {
      cv_score?: number | null;
      best_params?: Record<string, unknown>;
      error?: string;
    }
  >;
  best_model: {
    name: string;
    score: number;
  };
  evaluation: Record<string, unknown>;
  feature_importance?: { feature: string; importance: number }[] | null;
  download_url: string;
  message: string;
}

export interface ClusterProfile {
  top_numeric_signals: string[];
  dominant_categories?: string[];
}

export interface TaskConfig {
  task_type: TaskType;
  target_col?: string;
}

export interface ReviewConfig {
  choices: Record<string, unknown>;
  use_default_algorithms: boolean;
  selected_algorithms: string[];
  use_grid_search: boolean;
  param_mode: "default" | "custom";
  custom_params_text: string;
}

export interface AppState {
  session: UploadResponse | null;
  taskConfig: TaskConfig | null;
  analysis: AnalyzeResponse | null;
  review: ReviewConfig | null;
  training: TrainResponse | null;
}
