/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { PropsWithChildren } from "react";

import type { AnalyzeResponse, AppState, ReviewConfig, TaskConfig, TrainResponse, UploadResponse } from "../types/api";

interface AppStateContextType {
  state: AppState;
  setSession: (session: UploadResponse | null) => void;
  setTaskConfig: (taskConfig: TaskConfig | null) => void;
  setAnalysis: (analysis: AnalyzeResponse | null) => void;
  setReview: (review: ReviewConfig | null) => void;
  setTraining: (training: TrainResponse | null) => void;
  resetWorkflow: () => void;
}

const STORAGE_KEY = "automl_frontend_state_v1";

const defaultState: AppState = {
  session: null,
  taskConfig: null,
  analysis: null,
  review: null,
  training: null,
};

const AppStateContext = createContext<AppStateContextType | undefined>(undefined);

function loadInitialState(): AppState {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultState;
    return { ...defaultState, ...JSON.parse(raw) };
  } catch {
    return defaultState;
  }
}

export function AppStateProvider({ children }: PropsWithChildren) {
  const [state, setState] = useState<AppState>(() => loadInitialState());

  useEffect(() => {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [state]);

  const value = useMemo<AppStateContextType>(
    () => ({
      state,
      setSession: (session) => {
        setState((prev) => ({
          ...prev,
          session,
          taskConfig: session ? prev.taskConfig : null,
          analysis: session ? prev.analysis : null,
          review: session ? prev.review : null,
          training: session ? prev.training : null,
        }));
      },
      setTaskConfig: (taskConfig) => {
        setState((prev) => ({ ...prev, taskConfig }));
      },
      setAnalysis: (analysis) => {
        setState((prev) => ({ ...prev, analysis, training: analysis ? prev.training : null }));
      },
      setReview: (review) => {
        setState((prev) => ({ ...prev, review }));
      },
      setTraining: (training) => {
        setState((prev) => ({ ...prev, training }));
      },
      resetWorkflow: () => {
        setState(defaultState);
      },
    }),
    [state],
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState(): AppStateContextType {
  const context = useContext(AppStateContext);
  if (!context) {
    throw new Error("useAppState must be used inside AppStateProvider");
  }
  return context;
}
