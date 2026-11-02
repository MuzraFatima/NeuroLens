export type ProvenanceKind = "LIVE" | "CACHED" | "LOCAL STORE" | "SAMPLE / SYNTHETIC" | "UNAVAILABLE";
export interface ProvenanceBadgeSpec { kind: ProvenanceKind; source: string; version: string; }
