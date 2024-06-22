// src/components/chartConfig.ts
import { ApexOptions } from "apexcharts";
import featuresDriftData from "./testData";

const timepoints = Array.from({ length: 201 }, (_, i) => i);

const driftTypeMap: { [key: string]: number } = {
  linear_drift: 1,
  sudden_drift: 2,
  gradual_drift: 3,
};

export const featureNames = Object.keys(featuresDriftData);

const driftData = featureNames.map((feature) => {
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
});

const series = featureNames.map((feature, index) => ({
  name: feature,
  data: driftData[index].map((value, idx) => ({
    x: `T${idx}`,
    y: value !== null ? value : 0,
  })),
}));

const options: any = {
  chart: {
    height: 500,
    type: "heatmap",
  },
  plotOptions: {
    heatmap: {
      shadeIntensity: 0.5,
      radius: 0,
      useFillColorAsStroke: true,
      colorScale: {
        ranges: [
          {
            from: 1,
            to: 1,
            name: "Linear Drift",
            color: "#00A100",
          },
          {
            from: 2,
            to: 2,
            name: "Sudden Drift",
            color: "#128FD9",
          },
          {
            from: 3,
            to: 3,
            name: "Gradual Drift",
            color: "#FFB200",
          },
          {
            from: 0,
            to: 0,
            name: "No Drift",
            color: "#F3F4F5",
          },
        ],
      },
    },
  },
  dataLabels: {
    enabled: false,
  },
  stroke: {
    width: 1,
  },
  title: {
    text: "Feature Drift Over Time",
  },
  xaxis: {
    type: "category",
    categories: timepoints.map((tp) => `T${tp}`),
    title: {
      text: "Timepoints",
    },
  },
  yaxis: {
    categories: featureNames,
    title: {
      text: "Features",
    },
  },
};

export { series, options };
