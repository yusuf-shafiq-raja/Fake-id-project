import React, { useState } from "react";
import "./form.css";
import { predictInstagram } from "./api";

const Upload = () => {
  const [formData, setFormData] = useState({
    username: "",
    fullname: "",
    profilePic: "1",
    descriptionLength: "",
    externalURL: "0",
    private: "0",
    numPosts: "",
    numFollowers: "",
    numFollows: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    const processedData = {
      ...formData,
      numsLengthUsername:
        formData.username.replace(/\D/g, "").length / formData.username.length || 0,
      fullnameWords: formData.fullname.trim().split(/\s+/).length,
      numsLengthFullname:
        formData.fullname.replace(/\D/g, "").length / formData.fullname.length || 0,
      nameEqualsUsername:
        formData.username.toLowerCase() === formData.fullname.toLowerCase() ? 1 : 0,
    };

    console.log("Processed Data Sent to API:", processedData); // Debugging

    try {
      const prediction = await predictInstagram(processedData);
      setResult(prediction);
    } catch (error) {
      console.error("Error getting prediction:", error);
      setResult("Error fetching prediction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-container">
      <form onSubmit={handleSubmit} className="instagram-form">
        <h2>Instagram Fake Account Detection</h2>
        <input type="text" name="username" placeholder="Username" onChange={handleChange} required />
        <input type="text" name="fullname" placeholder="Full Name" onChange={handleChange} required />
        <input type="number" name="descriptionLength" placeholder="Bio Length" onChange={handleChange} required />
        <input type="number" name="numPosts" placeholder="Number of Posts" onChange={handleChange} required />
        <input type="number" name="numFollowers" placeholder="Followers" onChange={handleChange} required />
        <input type="number" name="numFollows" placeholder="Following" onChange={handleChange} required />
        <button type="submit" disabled={loading}>
          {loading ? "Checking..." : "Check Fake ID"}
        </button>
      </form>

      {result !== null && (
        <div className={`result-box ${result === "FAKE" ? "fake" : "real"}`}>
          <h3>Prediction Result:</h3>
          <p>{result === "FAKE" ? "🚨 This account is likely FAKE!" : "✅ This account seems REAL!"}</p>
        </div>
      )}
    </div>
  );
};

export default Upload;
