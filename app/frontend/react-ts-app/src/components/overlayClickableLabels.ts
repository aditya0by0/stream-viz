// src/components/overlayClickableLabels.ts
import { featureNames } from "./OverviewChartConfig";

export const addClickableLabels = () => {
  featureNames.forEach((feature: any, index: any) => {
    const label = document.querySelector(
      `.apexcharts-yaxis-labels text:nth-child(${index + 1})`
    );

    if (label) {
      const rect = label.getBoundingClientRect();
      const clickableDiv = document.createElement("div");

      clickableDiv.style.position = "absolute";
      clickableDiv.style.top = `${rect.top}px`;
      clickableDiv.style.left = `${rect.left}px`;
      clickableDiv.style.width = `${rect.width}px`;
      clickableDiv.style.height = `${rect.height}px`;
      clickableDiv.style.cursor = "pointer";
      clickableDiv.style.zIndex = "1000";
      clickableDiv.style.backgroundColor = "rgba(0, 0, 0, 0)"; // Transparent background
      clickableDiv.onclick = () => handleLabelClick(feature);

      document.body.appendChild(clickableDiv);
    }
  });
};

const handleLabelClick = (feature: string) => {
  fetch(`https://your-backend-api.com/feature-drift/${feature}`)
    .then((response) => response.json())
    .then((data) => console.log(data))
    .catch((error) => console.error("Error:", error));
};
