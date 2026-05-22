import { Link as RouterLink, useLocation, useNavigate } from "react-router-dom";
import {
  AppBar,
  Box,
  Button,
  Container,
  Link,
  Step,
  StepLabel,
  Stepper,
  Toolbar,
  Typography,
} from "@mui/material";
import type { PropsWithChildren } from "react";

import { useAppState } from "../context/AppStateContext";

const steps = [
  { label: "Upload", path: "/" },
  { label: "Configure", path: "/configure" },
  { label: "Review", path: "/review" },
  { label: "Results", path: "/results" },
];

function currentStep(pathname: string): number {
  const index = steps.findIndex((step) => step.path === pathname);
  return index >= 0 ? index : 0;
}

export function WizardLayout({ children }: PropsWithChildren) {
  const location = useLocation();
  const navigate = useNavigate();
  const { resetWorkflow } = useAppState();

  const handleReset = () => {
    resetWorkflow();
    navigate("/");
  };

  return (
    <Box sx={{ minHeight: "100vh", background: "linear-gradient(160deg, #f7f9fc 0%, #edf2ff 100%)" }}>
      <AppBar position="static" color="transparent" elevation={0}>
        <Toolbar sx={{ justifyContent: "space-between" }}>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Automated ML Frontend
          </Typography>
          <Button color="inherit" onClick={handleReset}>
            Reset Session
          </Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 3 }}>
        <Box sx={{ mb: 3 }}>
          <Stepper activeStep={currentStep(location.pathname)} alternativeLabel>
            {steps.map((step) => (
              <Step key={step.path}>
                <StepLabel>
                  <Link component={RouterLink} to={step.path} underline="hover" color="inherit">
                    {step.label}
                  </Link>
                </StepLabel>
              </Step>
            ))}
          </Stepper>
        </Box>

        {children}
      </Container>
    </Box>
  );
}
