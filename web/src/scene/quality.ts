export type QualityTier = "High" | "Medium" | "Low";
export interface TierConfig { dprCap: number; particles: number; surfaceSegments: number; bloom: boolean; }
export const QUALITY_TIERS: Record<QualityTier, TierConfig> = {
  High:   { dprCap: 2.0, particles: 20000, surfaceSegments: 256, bloom: true },
  Medium: { dprCap: 1.5, particles: 8000,  surfaceSegments: 128, bloom: false },
  Low:    { dprCap: 1.0, particles: 2000,  surfaceSegments: 64,  bloom: false },
};
