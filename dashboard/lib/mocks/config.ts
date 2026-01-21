/**
 * Mock configuration
 * Controls mock behavior globally
 */

export type MockScenario = "normal" | "empty" | "error" | "large";

export const mockConfig = {
  /** Enable mock data (always true in Phase 0) */
  useMockData: true,

  /** Simulated API latency in ms */
  simulateLatency: 800,

  /** Current scenario: normal, empty, error, large */
  scenario: "normal" as MockScenario,
};

/**
 * Simulate network delay
 */
export async function simulateDelay(ms?: number): Promise<void> {
  const delay = ms ?? mockConfig.simulateLatency;
  return new Promise((resolve) => setTimeout(resolve, delay));
}

/**
 * Throw error if scenario is 'error'
 */
export function checkMockError(): void {
  if (mockConfig.scenario === "error") {
    throw new Error("API unavailable (mock error)");
  }
}
