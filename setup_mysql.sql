
-- Run this in MySQL Workbench
CREATE DATABASE IF NOT EXISTS resume_screening;
USE resume_screening;

CREATE TABLE IF NOT EXISTS candidates (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    category     VARCHAR(100),
    clean_resume LONGTEXT,
    years_exp    INT DEFAULT 0,
    email        VARCHAR(200) DEFAULT "",
    word_count   INT DEFAULT 0,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_category (category)
);

CREATE TABLE IF NOT EXISTS jobs (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    title       VARCHAR(200),
    description LONGTEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS results (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    job_id       INT,
    candidate_id INT,
    match_score  DECIMAL(6,2),
    rank_pos     INT,
    screened_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id)
        REFERENCES jobs(id),
    FOREIGN KEY (candidate_id)
        REFERENCES candidates(id)
);

SELECT "Database ready!" AS status;
SHOW TABLES;
