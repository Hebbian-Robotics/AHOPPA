const recordingNamespace = new Uint8Array([
	0x6b, 0xa7, 0xb8, 0x11, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f,
	0xd4, 0x30, 0xc8,
]);

export async function videoUuid(recordingIdentifier: string): Promise<string> {
	// This namespace keeps existing shared recording links stable.
	const name = new TextEncoder().encode(
		`webolt:recording:${recordingIdentifier}`,
	);
	const input = new Uint8Array(recordingNamespace.length + name.length);
	input.set(recordingNamespace);
	input.set(name, recordingNamespace.length);
	const digest = await crypto.subtle.digest("SHA-1", input);
	const bytes = new Uint8Array(digest.slice(0, 16));
	bytes[6] = (bytes[6] & 0x0f) | 0x50;
	bytes[8] = (bytes[8] & 0x3f) | 0x80;
	const hexadecimal = Array.from(bytes, (byte) =>
		byte.toString(16).padStart(2, "0"),
	).join("");
	return [
		hexadecimal.slice(0, 8),
		hexadecimal.slice(8, 12),
		hexadecimal.slice(12, 16),
		hexadecimal.slice(16, 20),
		hexadecimal.slice(20),
	].join("-");
}
