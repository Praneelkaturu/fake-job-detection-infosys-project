import { useState } from "react";

export default function App() {
  const [jobText, setJobText] = useState("");
  const [company, setCompany] = useState("");
  const [location, setLocation] = useState("");
  const [salary, setSalary] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [flagged, setFlagged] = useState(false);
  const [flagReason, setFlagReason] = useState("");
  const [comments, setComments] = useState("");
  const [email, setEmail] = useState("");

  const API_BASE = "http://localhost:8000"; // change if deployed

  const handleSubmit = async () => {
    if (!jobText.trim()) {
      setError("Job description cannot be empty");
      return;
    }
    setError("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: jobText,
          company,
          location,
          salary
        })
      });

      if (!res.ok) throw new Error("Server error");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError("Unable to process request. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setJobText("");
    setCompany("");
    setLocation("");
    setSalary("");
    setResult(null);
    setError("");
    setFlagged(false);
  };

  const handleFlag = async () => {
    await fetch(`${API_BASE}/flag`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_text: jobText,
        prediction: result?.classification,
        reason: flagReason,
        comments,
        email
      })
    });
    setFlagged(true);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6 flex justify-center">
      <div className="max-w-3xl w-full bg-white p-6 rounded-2xl shadow">
        <h1 className="text-2xl font-bold mb-4">Fake Job Post Detection</h1>

        <textarea
          className="w-full border p-3 rounded mb-2"
          rows={6}
          placeholder="Paste job description here"
          value={jobText}
          onChange={(e) => setJobText(e.target.value)}
        />
        <div className="text-sm text-gray-500 mb-3">
          Characters: {jobText.length}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          <input className="border p-2 rounded" placeholder="Company (optional)" value={company} onChange={(e) => setCompany(e.target.value)} />
          <input className="border p-2 rounded" placeholder="Location (optional)" value={location} onChange={(e) => setLocation(e.target.value)} />
          <input className="border p-2 rounded" placeholder="Salary (optional)" value={salary} onChange={(e) => setSalary(e.target.value)} />
        </div>

        <div className="flex gap-3">
          <button onClick={handleSubmit} className="bg-blue-600 text-white px-4 py-2 rounded">
            Analyze Job
          </button>
          <button onClick={handleReset} className="bg-gray-300 px-4 py-2 rounded">
            Clear
          </button>
        </div>

        {loading && <p className="mt-4">Processing...</p>}
        {error && <p className="mt-4 text-red-600">{error}</p>}

        {result && (
          <div className="mt-6 p-4 border rounded">
            <h2 className="text-xl font-semibold">Result</h2>
            <p className={`text-lg font-bold ${result.classification === "Real" ? "text-green-600" : "text-red-600"}`}>
              {result.classification}
            </p>
            <p>Confidence: {result.confidence}%</p>
            <p>Processing Time: {result.processing_time} ms</p>
            <p className="text-sm text-gray-500">{result.timestamp}</p>

            {result.confidence < 60 && (
              <p className="text-yellow-600 mt-2">⚠ Low confidence prediction</p>
            )}

            {!flagged && (
              <div className="mt-4">
                <h3 className="font-semibold">Flag this post</h3>
                <select className="border p-2 w-full mt-2" onChange={(e) => setFlagReason(e.target.value)}>
                  <option value="">Select reason</option>
                  <option value="Scam">Scam</option>
                  <option value="Fake recruiter">Fake recruiter</option>
                  <option value="Suspicious salary">Suspicious salary</option>
                </select>
                <textarea className="border p-2 w-full mt-2" placeholder="Additional comments" onChange={(e) => setComments(e.target.value)} />
                <input className="border p-2 w-full mt-2" placeholder="Email (optional)" onChange={(e) => setEmail(e.target.value)} />
                <button onClick={handleFlag} className="mt-3 bg-red-600 text-white px-4 py-2 rounded">Flag</button>
              </div>
            )}

            {flagged && <p className="text-green-600 mt-3">Thank you. Post flagged successfully.</p>}
          </div>
        )}
      </div>
    </div>
  );
}
