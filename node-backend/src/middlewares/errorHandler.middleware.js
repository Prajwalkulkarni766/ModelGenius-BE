import { ApiError } from "../utils/ApiError.js";
import logger, { maskSensitiveFields } from "../utils/logger.js";

const errorHandler = (err, req, res, next) => {
  let success = err.success || false;
  let statusCode = err.statusCode || 500;
  let message = err.message || "Internal Server Error";
  let errors = err.errors || [];

  if (!(err instanceof ApiError)) {
    success = false;
    statusCode = 500;
    message = "Internal Server Error";
  }

  logger.error("Request error", {
    method: req.method,
    url: req.originalUrl,
    status: statusCode,
    message,
    errors,
    stack: process.env.NODE_ENV === "development" ? err.stack : undefined,
    payload: req.body && Object.keys(req.body).length > 0 
      ? maskSensitiveFields(req.body) 
      : undefined,
  });

  return res.status(statusCode).json({
    success: false,
    statusCode,
    message,
    errors,
    data: null,
    ...(process.env.NODE_ENV === "development" && { stack: err.stack }),
  });
};

export { errorHandler };
