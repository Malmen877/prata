//! Runs the real prata-web binary with a fake KLANG_API_KEY against a local mock Klang API
//! and checks that the key never shows up in its stdout/stderr (nor in the HTTP responses),
//! while the startup and sync lines do.

use std::{
    process::{Command, Stdio},
    time::Duration,
};

use axum::{extract::Path, http::HeaderMap, http::StatusCode, response::IntoResponse, routing::get, Router};

const KEY: &str = "sk_fake_log_test_7f3c9e1d2b";

async fn mock() -> String {
    let app = Router::new()
        .route("/api/v1/conversations", get(|| async {
            ([("content-type", "application/json")],
             r#"{"data":[{"id":"ok1","title":null,"status":"ready","created_at":"2026-09-30T08:51:00Z","updated_at":"2026-09-30T09:00:00Z","sources":[]},
                         {"id":"bad2","title":"x","status":"ready","created_at":"2026-09-30T08:51:00Z","updated_at":"2026-09-30T09:00:00Z","sources":[]}],
                "has_more":false,"next_cursor":null}"#)
        }))
        .route("/api/v1/conversations/{id}", get(|Path(id): Path<String>, h: HeaderMap| async move {
            if id == "ok1" {
                return ([("content-type", "application/json")],
                        r###"{"id":"ok1","status":"ready","created_at":"2026-09-30T08:51:00Z","updated_at":"2026-09-30T09:00:00Z","summary":"## Loggtest","sources":[{"type":"transcript","content":"Talare 1: Hej."}]}"###.to_string()).into_response();
            }
            // a hostile upstream echoing the credentials back – must not reach the logs
            let auth = h.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
            (StatusCode::UNAUTHORIZED, format!(r#"{{"error":{{"type":"auth","message":"bad key {auth}"}}}}"#)).into_response()
        }));
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    format!("http://{addr}/api/v1")
}

#[tokio::test]
async fn key_never_reaches_stdout_or_stderr() {
    let base = mock().await;
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("out.log");
    let err_path = tmp.path().join("err.log");
    let mut child = Command::new(env!("CARGO_BIN_EXE_prata-web"))
        .env("KLANG_API_KEY", KEY)
        .env("PRATA_KLANG_BASE_URL", &base)
        .env("PRATA_WEB_PORT", port.to_string())
        .env("PRATA_WEB_HOST", "127.0.0.1")
        .env("PRATA_NOTES_DIR", tmp.path().join("notes"))
        .env("PRATA_WORK_DIR", tmp.path().join("work"))
        .env("PRATA_BIN", "/nonexistent/prata")
        .stdin(Stdio::null())
        // plain files, like launchd's StandardOutPath: lines must arrive without a tty
        .stdout(std::fs::File::create(&out_path).unwrap())
        .stderr(std::fs::File::create(&err_path).unwrap())
        .spawn()
        .unwrap();
    let http = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}");
    let mut up = false;
    for _ in 0..100 {
        if http.get(format!("{url}/api/health")).send().await.is_ok() {
            up = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(up, "server did not start: {}", std::fs::read_to_string(&err_path).unwrap_or_default());
    let mut bodies = String::new();
    bodies += &http.get(format!("{url}/api/info")).send().await.unwrap().text().await.unwrap();
    bodies += &http.post(format!("{url}/api/klang/sync")).send().await.unwrap().text().await.unwrap();
    let mut done = false;
    for _ in 0..100 {
        let s = http.get(format!("{url}/api/klang/sync")).send().await.unwrap().text().await.unwrap();
        bodies += &s;
        if s.contains("\"running\":false") {
            done = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    bodies += &http.get(format!("{url}/api/notes")).send().await.unwrap().text().await.unwrap();
    // the logs are written line by line: the sync line is there while the server still runs
    tokio::time::sleep(Duration::from_millis(200)).await;
    let logs_running = std::fs::read_to_string(&out_path).unwrap() + &std::fs::read_to_string(&err_path).unwrap();
    let _ = child.kill();
    let _ = child.wait();
    let logs = std::fs::read_to_string(&out_path).unwrap() + &std::fs::read_to_string(&err_path).unwrap();
    assert!(done, "sync did not finish: {bodies}");
    assert!(bodies.contains("\"klang_enabled\":true"), "{bodies}");
    assert!(bodies.contains("Ogiltig Klang-nyckel"), "{bodies}");
    assert!(bodies.contains("Loggtest"), "imported note listed: {bodies}");
    assert!(logs_running.contains("klang: import enabled"), "{logs_running}");
    assert!(logs_running.contains("[klang] Ogiltig Klang-nyckel"), "{logs_running}");
    for (what, text) in [("stdout/stderr", &logs), ("HTTP responses", &bodies)] {
        assert!(!text.contains(KEY), "API key leaked into {what}:\n{text}");
        assert!(!text.contains("sk_fake_log"), "part of the API key leaked into {what}");
    }
    // the notes on disk don't carry it either
    for e in walkdir(&tmp.path().join("notes")) {
        assert!(!std::fs::read_to_string(&e).unwrap_or_default().contains(KEY), "{}", e.display());
    }
}

fn walkdir(d: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut v = vec![];
    for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() { v.extend(walkdir(&p)) } else { v.push(p) }
    }
    v
}
