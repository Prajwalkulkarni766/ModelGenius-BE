import { createLogger, format, transports } from "winston";
import winstonDaily from "winston-daily-rotate-file";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const { combine, timestamp, json, colorize, simple } = format;
const isProduction = process.env.NODE_ENV === "production";

const logDir = process.env.LOG_DIR || path.join(__dirname, "../../logs");

const dailyRotateTransport = new winstonDaily({
  level: process.env.LOG_LEVEL || "info",
  filename: "app-%DATE%.log",
  datePattern: "YYYY-MM-DD",
  zippedArchive: true,
  maxSize: "10m",
  maxFiles: "7d",
  dirname: logDir,
  format: combine(timestamp(), json()),
});

const consoleTransport = new transports.Console({
  level: process.env.LOG_LEVEL || (isProduction ? "info" : "debug"),
  format: isProduction
    ? combine(timestamp(), json())
    : combine(colorize(), timestamp(), simple()),
});

const logger = createLogger({
  level: process.env.LOG_LEVEL || "info",
  format: combine(timestamp(), json()),
  transports: [consoleTransport, dailyRotateTransport],
});

export default logger;

export function maskSensitiveFields(data) {
  if (!data || typeof data !== "object") return data;

  const sensitiveFields = [
    "password",
    "refreshToken",
    "accessToken",
    "token",
    "secret",
    "apiKey",
    "authorization",
    "cookie",
  ];

  const masked = Array.isArray(data) ? [] : {};
  for (const [key, value] of Object.entries(data)) {
    const lowerKey = key.toLowerCase();
    if (sensitiveFields.some((f) => lowerKey.includes(f))) {
      masked[key] = "[REDACTED]";
    } else if (typeof value === "object" && value !== null) {
      masked[key] = maskSensitiveFields(value);
    } else {
      masked[key] = value;
    }
  }
  return masked;
}