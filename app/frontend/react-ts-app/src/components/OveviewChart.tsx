import React, { useEffect } from "react";
import ReactApexChart from "react-apexcharts";
import { Box } from "@mui/material";
import { series, options } from "./OverviewChartConfig";
import { addClickableLabels } from "./overlayClickableLabels";

const OverviewChart: React.FC = () => {
  useEffect(() => {
    addClickableLabels();
  }, []);

  return (
    <Box sx={{ width: "100%", height: "500px" }}>
      <ReactApexChart
        options={options}
        series={series}
        type="heatmap"
        height={300}
      />
    </Box>
  );
};

export default OverviewChart;
