import assert from "node:assert/strict";
import { test } from "node:test";
import { videoUuid } from "./videoIdentifiers.ts";

test("recordings have stable, distinct UUIDs independent of host and page reload", async () => {
	// UUIDv5 values independently computed with Python uuid.uuid5(NAMESPACE_URL, name).
	assert.equal(
		await videoUuid("isaac-integration-sol-v2"),
		"c05b0d6a-dd83-5aea-bc1a-8baf09788143",
	);
	assert.equal(
		await videoUuid("isaac-integration-gemini-v2"),
		"6b0edb31-f8a3-5e74-b6d4-4f12fc68e693",
	);
});
