interface DriftRange {
  start_tp: number;
  end_tp: number;
}

interface Drift {
  linear_drift?: DriftRange;
  sudden_drift?: DriftRange;
  gradual_drift?: DriftRange;
}

interface FeaturesDriftData {
  [key: string]: Drift[];
}

const featuresDriftData: FeaturesDriftData = {
  feature_1: [
    { linear_drift: { start_tp: 30, end_tp: 40 } },
    { sudden_drift: { start_tp: 100, end_tp: 110 } },
    { gradual_drift: { start_tp: 150, end_tp: 160 } },
  ],
  feature_2: [
    { linear_drift: { start_tp: 50, end_tp: 60 } },
    { sudden_drift: { start_tp: 120, end_tp: 130 } },
  ],
  feature_3: [{ gradual_drift: { start_tp: 70, end_tp: 80 } }],
};

export default featuresDriftData;
