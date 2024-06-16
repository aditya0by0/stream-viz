// src/components/FeatureDriftChart.tsx

import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { Box, Typography } from "@mui/material";
import featuresDriftData from "./testData";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const timepoints = Array.from({ length: 201 }, (_, i) => i);
const driftTypeMap: { [key: string]: number } = {
  linear_drift: 1,
  sudden_drift: 2,
  gradual_drift: 3,
};

const FeatureDriftChart: React.FC = () => {
  const driftData: number[][] = Object.keys(featuresDriftData).map(
    (feature) => {
      const driftArray = new Array(201).fill(null);
      featuresDriftData[feature].forEach((drift) => {
        Object.entries(drift).forEach(([driftType, timeRange]) => {
          if (timeRange) {
            const start = timeRange.start_tp;
            const end = timeRange.end_tp;
            for (let i = start; i <= end; i++) {
              driftArray[i] = driftTypeMap[driftType];
            }
          }
        });
      });
      return driftArray;
    }
  );

  const data = {
    labels: timepoints,
    datasets: Object.keys(featuresDriftData).map((feature, index) => ({
      label: feature,
      data: driftData[index],
      borderColor: `rgba(${(index + 1) * 50}, 99, 132, 0.6)`,
      backgroundColor: `rgba(${(index + 1) * 50}, 99, 132, 0.2)`,
      fill: false,
      spanGaps: true,
    })),
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: "Feature Drift Over Time",
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1,
          callback: (value: number | string) => {
            switch (value) {
              case 1:
                return "Linear Drift";
              case 2:
                return "Sudden Drift";
              case 3:
                return "Gradual Drift";
              default:
                return "";
            }
          },
        },
      },
      x: {
        title: {
          display: true,
          text: "Timepoints",
        },
      },
    },
  };

  return (
    <Box sx={{ width: "100%", height: "500px" }}>
      <Line data={data} options={options} />
    </Box>
  );
};

export default FeatureDriftChart;
