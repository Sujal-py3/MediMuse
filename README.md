# MediMuse 🩺💬  
![LLM-powered](https://img.shields.io/badge/LLM-powered-blueviolet)  
![Hackathon Winner](https://img.shields.io/badge/Hackathon-6th%20Place%20out%20of%2012%2C000+-red)  
![MongoDB](https://img.shields.io/badge/Database-MongoDB-green)  
![Redis](https://img.shields.io/badge/Caching-Redis-informational)

> 🧠 An AI-powered healthcare assistant for managing patient data, scheduling, and intelligent query handling using LLMs.

---

## 🧩 Overview
**MediMuse** is a full-stack AI chatbot designed for healthcare professionals and institutions. It streamlines patient record management, answers natural language queries about medications, and predicts potential health risks using structured patient data.  
The project was built and demoed as part of a competitive hackathon, where it ranked **6th out of 12,000+** participants.

---

## 🛠️ Tech Stack
- **Frontend:** React.js (Patient dashboard, chat interface)
- **Backend:** Node.js + Express
- **Database:** MongoDB (HIPAA-styled schemas), Redis (for caching + performance)
- **Auth:** JWT, OAuth2
- **LLMs:** TogetherAI API for chat + predictive tasks
- **Misc:** Python for analytics module, Dockerized microservices

---

## 🧑‍💻 My Role & Contributions
- Architected the backend microservices in Node.js and designed MongoDB schemas to manage 5,000+ patient records.
- Integrated Redis for caching & throttling, improving API performance by **40%**.
- Developed Python modules for early health risk detection based on patient history.
- Secured all user sessions with JWT-based authentication.
- Designed a clean React-based UI and built a multi-intent chatbot experience using LLMs.
- Optimized LLM prompts for medical context comprehension (e.g., prescriptions, symptoms).

---

## 📸 Snapshots

<p align="center">
  <img src="./assets/medimuse_dashboard.png" width="48%" />
  <img src="./assets/medimuse_chat.png" width="48%" />
</p>

---

## 📁 Folder Structure

```bash
medimuse/
├── backend/
│   ├── controllers/
│   ├── routes/
│   └── utils/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
├── python-analytics/
│   └── predict_risks.py
├── .env.example
├── README.md
