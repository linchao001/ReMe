export const BRAND_NAME = "Qifeng";
export const APP_TITLE = "Qifeng ZHB";

/** Replace ReMe branding in config JSON shown in settings without mutating source data. */
export function formatConfigForDisplay(config: unknown): string {
  return JSON.stringify(config, null, 2).replace(/\bReMe\b/g, BRAND_NAME);
}
