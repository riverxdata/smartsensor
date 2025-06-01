import React, { useState } from "react";
import { FiUploadCloud } from "react-icons/fi";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult("");
  };

  const handlePredict = () => {
    if (!file) {
      setResult("Please upload a CSV file.");
      return;
    }
    setLoading(true);
    // Giả lập kết quả dự đoán
    setTimeout(() => {
      setResult("The output is [2]");
      setLoading(false);
    }, 1200);
  };

  return (
    <div className="container upgraded">
      <h1>
        <span className="iris-gradient">smartSensor</span>
      </h1>
      <div className="banner upgraded-banner">
        <span>
          <FiUploadCloud size={38} style={{ marginRight: 12, verticalAlign: 'middle', color: '#ff3e81' }} />
          Streamlit Iris Flower Classifier ML App
        </span>
      </div>
      <div className="upload-section upgraded-upload">
        <label htmlFor="file-upload" className="upload-label">
          <FiUploadCloud size={22} style={{ marginRight: 6, verticalAlign: 'middle' }} />
          Upload CSV file
        </label>
        <input
          id="file-upload"
          type="file"
          accept=".csv"
          onChange={handleFileChange}
        />
        <div className="upload-hint">Chọn file CSV chứa dữ liệu hoa Iris để dự đoán. File phải có đúng định dạng.</div>
        {file && <div className="file-name">Đã chọn: <b>{file.name}</b></div>}
      </div>
      <button className="predict-btn upgraded-btn" onClick={handlePredict} disabled={loading}>
        {loading ? (
          <span className="loader"></span>
        ) : (
          <>
            <FiUploadCloud size={18} style={{ marginRight: 7, verticalAlign: 'middle' }} /> Predict
          </>
        )}
      </button>
      {result && (
        <div className={result.includes("output") ? "result success upgraded-result" : "result error upgraded-result"}>{result}</div>
      )}
    </div>
  );
}
