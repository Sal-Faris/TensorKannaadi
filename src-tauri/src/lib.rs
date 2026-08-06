use rand::{distributions::Alphanumeric, Rng};
use serde::Serialize;
use std::{
    env, fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{mpsc, Mutex},
    thread,
    time::Duration,
};
use tauri::{Emitter, Manager, RunEvent, State};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SidecarInfo {
    base_url: String,
    token: String,
    state: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct WorkspaceEntry {
    name: String,
}

struct ManagedSidecar {
    info: Mutex<SidecarInfo>,
    child: Mutex<Option<Child>>,
}

impl ManagedSidecar {
    fn new() -> Self {
        Self {
            info: Mutex::new(SidecarInfo {
                base_url: String::new(),
                token: String::new(),
                state: "starting".into(),
            }),
            child: Mutex::new(None),
        }
    }

    fn stop(&self) {
        if let Ok(mut child_guard) = self.child.lock() {
            if let Some(child) = child_guard.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
            *child_guard = None;
        }
        if let Ok(mut info) = self.info.lock() {
            info.state = "stopped".into();
        }
    }
}

#[tauri::command]
fn sidecar_info(state: State<'_, ManagedSidecar>) -> Result<SidecarInfo, String> {
    state
        .info
        .lock()
        .map(|info| info.clone())
        .map_err(|_| "Sidecar state is unavailable".into())
}

fn safe_workspace_name(name: &str) -> Result<String, String> {
    let value: String = name
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | ' ') {
                character
            } else {
                '-'
            }
        })
        .collect();
    let value = value.trim_matches([' ', '-', '_']).to_string();
    if value.is_empty() {
        Err("Workspace name must contain letters or numbers".into())
    } else {
        Ok(value.chars().take(80).collect())
    }
}

fn workspace_directory(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let directory = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Application data directory is unavailable: {error}"))?
        .join("workspaces");
    fs::create_dir_all(&directory)
        .map_err(|error| format!("Workspace directory could not be created: {error}"))?;
    Ok(directory)
}

#[tauri::command]
fn save_workspace(
    app: tauri::AppHandle,
    name: String,
    contents: String,
) -> Result<WorkspaceEntry, String> {
    let name = safe_workspace_name(&name)?;
    let path = workspace_directory(&app)?.join(format!("{name}.json"));
    fs::write(&path, contents).map_err(|error| format!("Workspace could not be saved: {error}"))?;
    Ok(WorkspaceEntry { name })
}

#[tauri::command]
fn list_workspaces(app: tauri::AppHandle) -> Result<Vec<WorkspaceEntry>, String> {
    let mut entries = Vec::new();
    for item in fs::read_dir(workspace_directory(&app)?)
        .map_err(|error| format!("Workspaces could not be listed: {error}"))?
    {
        let item = item.map_err(|error| format!("A workspace entry could not be read: {error}"))?;
        let path = item.path();
        if path.extension().and_then(|value| value.to_str()) == Some("json") {
            if let Some(name) = path.file_stem().and_then(|value| value.to_str()) {
                entries.push(WorkspaceEntry { name: name.into() });
            }
        }
    }
    entries.sort_by_key(|entry| entry.name.to_lowercase());
    Ok(entries)
}

#[tauri::command]
fn load_workspace(app: tauri::AppHandle, name: String) -> Result<String, String> {
    let name = safe_workspace_name(&name)?;
    let path = workspace_directory(&app)?.join(format!("{name}.json"));
    fs::read_to_string(path).map_err(|error| format!("Workspace could not be opened: {error}"))
}

fn backend_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("backend")
}

fn python_executable(backend: &Path) -> String {
    if let Ok(configured) = env::var("KANNAADI_PYTHON") {
        return configured;
    }
    let local = if cfg!(target_os = "windows") {
        backend.join(".venv").join("Scripts").join("python.exe")
    } else {
        backend.join(".venv").join("bin").join("python")
    };
    if local.exists() {
        local.to_string_lossy().into_owned()
    } else if cfg!(target_os = "windows") {
        "python.exe".into()
    } else {
        "python3".into()
    }
}

fn start_sidecar(app: tauri::AppHandle) -> Result<(), String> {
    let token: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(48)
        .map(char::from)
        .collect();
    let backend = backend_directory()
        .canonicalize()
        .map_err(|error| format!("Backend directory is unavailable: {error}"))?;
    let python = python_executable(&backend);
    let mut command = Command::new(&python);
    command
        .current_dir(&backend)
        .env("PYTHONPATH", &backend)
        .args([
            "-m",
            "kannaadi.sidecar",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--token",
            &token,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);

    let mut child = command
        .spawn()
        .map_err(|error| format!("Could not launch Python at '{python}': {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or("Python sidecar stdout was not captured")?;
    let stderr = child.stderr.take();
    let (ready_tx, ready_rx) = mpsc::channel::<u16>();
    let output_handle = app.clone();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Some(value) = line.strip_prefix("KANNAADI_READY:") {
                if let Ok(port) = value.parse::<u16>() {
                    let _ = ready_tx.send(port);
                    continue;
                }
            }
            let _ = output_handle.emit("sidecar-log", line);
        }
    });
    if let Some(output) = stderr {
        let handle = app.clone();
        thread::spawn(move || {
            for line in BufReader::new(output).lines().map_while(Result::ok) {
                let _ = handle.emit("sidecar-log", line);
            }
        });
    }

    let port = match ready_rx.recv_timeout(Duration::from_secs(15)) {
        Ok(port) => port,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "Python sidecar did not reserve a loopback port: {error}"
            ));
        }
    };
    let state = app.state::<ManagedSidecar>();
    {
        let mut info = state.info.lock().map_err(|_| "Sidecar state lock failed")?;
        *info = SidecarInfo {
            base_url: format!("http://127.0.0.1:{port}"),
            token,
            state: "ready".into(),
        };
    }
    *state
        .child
        .lock()
        .map_err(|_| "Sidecar process lock failed")? = Some(child);
    let _ = app.emit("sidecar-status", "ready");

    let monitor = app.clone();
    thread::spawn(move || loop {
        thread::sleep(Duration::from_millis(350));
        let state = monitor.state::<ManagedSidecar>();
        let exited = state.child.lock().ok().and_then(|mut guard| {
            guard
                .as_mut()
                .and_then(|child| child.try_wait().ok().flatten())
        });
        if let Some(status) = exited {
            if let Ok(mut info) = state.info.lock() {
                info.state = if status.success() {
                    "stopped".into()
                } else {
                    "crashed".into()
                };
            }
            let _ = monitor.emit(
                "sidecar-status",
                if status.success() {
                    "stopped"
                } else {
                    "crashed"
                },
            );
            break;
        }
    });
    Ok(())
}

pub fn run() {
    let app = tauri::Builder::default()
        .manage(ManagedSidecar::new())
        .invoke_handler(tauri::generate_handler![
            sidecar_info,
            save_workspace,
            list_workspaces,
            load_workspace
        ])
        .setup(|app| {
            if let Err(error) = start_sidecar(app.handle().clone()) {
                if let Ok(mut info) = app.state::<ManagedSidecar>().info.lock() {
                    info.state = format!("error: {error}");
                }
                let _ = app.emit("sidecar-status", format!("error: {error}"));
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build the Kannaadi desktop application");

    app.run(|handle, event| {
        if matches!(event, RunEvent::Exit | RunEvent::ExitRequested { .. }) {
            handle.state::<ManagedSidecar>().stop();
        }
    });
}
