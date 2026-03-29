import mongoose from "mongoose";
import dotenv from "dotenv";
import bcrypt from "bcrypt";

import { User } from "../src/models/user.model.js";
import { Project } from "../src/models/project.model.js";
import { Dataset } from "../src/models/dataset.model.js";
import { Model } from "../src/models/model.model.js";
import { Notification } from "../src/models/notification.model.js";

dotenv.config();

const MONGO_URI = "mongodb://localhost:27017/modelgenius";

async function seedDatabase() {
  try {
    await mongoose.connect(MONGO_URI);
    console.log("MongoDB Connected");

    // Clear old data
    // await Promise.all([
    //   User.deleteMany(),
    //   Project.deleteMany(),
    //   Dataset.deleteMany(),
    //   Model.deleteMany(),
    //   Notification.deleteMany()
    // ]);

    // console.log("Old records removed");

    /* ============================
       1️⃣ USERS (25)
    ============================ */

    const indianUsers = [
      "Aarav Sharma", "Vivaan Patel", "Aditya Verma", "Vihaan Reddy", "Arjun Mehta",
      "Sai Kiran", "Rohan Kulkarni", "Yash Deshmukh", "Karthik Iyer", "Manish Nair",
      "Ananya Joshi", "Diya Shah", "Ishita Rao", "Sneha Pillai", "Pooja Gupta",
      "Kavya Menon", "Ritika Singh", "Priya Choudhary", "Neha Agarwal", "Shruti Mishra",
      "Rahul Tiwari", "Siddharth Bansal", "Harsh Vyas", "Abhishek Yadav", "Nikhil Jain"
    ];

    let password = await bcrypt.hash("Password@123", 10);

    const users = await User.insertMany(
      indianUsers.map((name, index) => ({
        username: name.toLowerCase().replace(/ /g, "_"),
        email: `${name.toLowerCase().replace(/ /g, ".")}@gmail.com`,
        password: password
      }))
    );

    console.log("Users inserted");

    /* ============================
       2️⃣ PROJECTS (25)
    ============================ */

    const projects = await Project.insertMany(
      users.map((user, index) => ({
        projectTitle: `ML Project ${index + 1} - Customer Insights`,
        projectDescription: `Machine Learning project for predictive analytics module ${index + 1}`,
        userId: user._id
      }))
    );

    console.log("Projects inserted");

    /* ============================
       3️⃣ DATASETS (25)
    ============================ */

    const datasets = await Dataset.insertMany(
      projects.map((project, index) => ({
        datasetFilePath: `/datasets/project_${index + 1}.csv`,
        originalFileName: `customer_data_${index + 1}.csv`,
        fileSize: 1024 * (index + 10),
        columns: [
          "customer_id",
          "age",
          "income",
          "gender",
          "purchase_history",
          "target"
        ],
        projectId: project._id
      }))
    );

    console.log("Datasets inserted");

    /* ============================
       4️⃣ MODELS (25)
    ============================ */

    const algorithms = [
      "logistic",
      "knn",
      "svm",
      "random_forest",
      "gradient_boosting",
      "linear_regression"
    ];

    const models = await Model.insertMany(
      datasets.map((dataset, index) => ({
        modelName: `Model_${index + 1}`,
        datasetId: dataset._id,
        targetColumn: "target",
        handlingMissingValueStrategy: "mean",
        encodingCategoricalMethod: "one_hot",
        normalizationTechnique: "zscore",
        algorithm: algorithms[index % algorithms.length],
        metrics: {
          accuracy: (0.75 + Math.random() * 0.2).toFixed(3),
          precision: (0.70 + Math.random() * 0.2).toFixed(3),
          recall: (0.72 + Math.random() * 0.2).toFixed(3),
          f1_score: (0.74 + Math.random() * 0.2).toFixed(3)
        },
        modelPath: `/models/model_${index + 1}.pkl`,
        projectId: projects[index]._id,
        userId: users[index]._id
      }))
    );

    console.log("Models inserted");

    /* ============================
       5️⃣ NOTIFICATIONS (25)
    ============================ */

    const severities = ["info", "success", "warning", "error"];

    await Notification.insertMany(
      users.map((user, index) => ({
        severity: severities[index % severities.length],
        title: `Training Update - Project ${index + 1}`,
        description: `Model training completed successfully with accuracy above threshold.`,
        userId: user._id
      }))
    );

    console.log("Notifications inserted");

    console.log("Database Seeding Completed Successfully");
    process.exit(0);

  } catch (error) {
    console.error("Seeding Failed:", error);
    process.exit(1);
  }
}

seedDatabase();