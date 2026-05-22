import { Navigate, Route, Routes } from "react-router-dom";

import { WizardLayout } from "./components/WizardLayout";
import { ConfigurePage } from "./pages/ConfigurePage";
import { ResultsPage } from "./pages/ResultsPage";
import { ReviewPage } from "./pages/ReviewPage";
import { UploadPage } from "./pages/UploadPage";

function App() {
  return (
    <WizardLayout>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/configure" element={<ConfigurePage />} />
        <Route path="/review" element={<ReviewPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </WizardLayout>
  );
}

export default App;
