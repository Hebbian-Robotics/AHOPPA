export async function copyVideoIdentifier(
	identifier: string,
): Promise<boolean> {
	try {
		await navigator.clipboard.writeText(identifier);
		return true;
	} catch {
		const previousFocus = document.activeElement;
		const selection = document.createElement("textarea");
		selection.value = identifier;
		selection.readOnly = true;
		selection.style.position = "fixed";
		selection.style.opacity = "0";
		document.body.append(selection);
		selection.select();
		try {
			return document.execCommand("copy");
		} finally {
			selection.remove();
			if (previousFocus instanceof HTMLElement) previousFocus.focus();
		}
	}
}
