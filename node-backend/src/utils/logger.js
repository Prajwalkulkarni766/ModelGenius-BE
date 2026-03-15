import { createLogger, format, transports } from "winston";
const { combine, timestamp, json, colorize, simple } = format;
const isProduction = process.env.NODE_ENV === "production";
const logger = createLogger({
  level: process.env.LOG_LEVEL || (isProduction ? "info" : "debug"),
  format: isProduction
    ? combine(timestamp(), json())
    : combine(colorize(), timestamp(), simple()),
  transports: [new transports.Console()],
});
export default logger;
