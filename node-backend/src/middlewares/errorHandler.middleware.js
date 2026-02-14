import { ApiError } from "../utils/ApiError.js";

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
