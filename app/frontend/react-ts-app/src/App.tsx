import React from "react";
import logo from "./logo.svg";
import "./App.css";
import { Container } from "@mui/material";
import OverviewChart from "./components/OveviewChart";
import Typography from "@mui/material";

function App() {
  return (
    <Container>
      <h3>Feature Drift Visualization</h3>
      <OverviewChart />
    </Container>
  );
}

export default App;
