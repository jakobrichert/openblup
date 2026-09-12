//! End-to-end tests for the `openblup` binary.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_file(name: &str, content: &str) -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "openblup_cli_{}_{}_{}",
        std::process::id(),
        id,
        name
    ));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

fn openblup() -> Command {
    Command::new(env!("CARGO_BIN_EXE_openblup"))
}

/// Balanced trial: 3 genotypes x 2 reps, rep coded numerically, one missing plot.
const TRIAL_CSV: &str = "genotype,rep,yield\n\
G1,1,10.0\n\
G2,1,8.0\n\
G3,1,6.0\n\
G1,2,12.0\n\
G2,2,10.0\n\
G3,2,8.0\n\
G1,3,NA\n";

/// Mrode (2005) Example 3.1.
const MRODE_PED: &str = "animal,sire,dam\n1,0,0\n2,0,0\n3,0,0\n4,1,0\n5,3,2\n6,1,2\n7,4,5\n8,3,6\n";
const MRODE_DATA: &str =
    "animal,sex,pwg\n4,male,4.5\n5,female,2.9\n6,female,3.9\n7,male,3.5\n8,male,5.0\n";

#[test]
fn fit_text_output_with_numeric_factor_and_missing_response() {
    let data = temp_file("trial.csv", TRIAL_CSV);
    let out = openblup()
        .args([
            "fit",
            "--data",
            data.to_str().unwrap(),
            "--response",
            "yield",
            "--fixed",
            "mu + rep",
            "--factor",
            "rep",
            "--random",
            "genotype",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {}", stderr);
    assert!(
        stderr.contains("Dropped 1 rows with missing 'yield'"),
        "{}",
        stderr
    );
    assert!(stdout.contains("Variance Components"), "{}", stdout);
    assert!(stdout.contains("genotype"), "{}", stdout);
    // treatment contrasts: intercept + rep.2 (rep 1 is the reference)
    assert!(stdout.contains("mu.intercept"), "{}", stdout);
    assert!(stdout.contains("rep.2"), "{}", stdout);
    assert!(!stdout.contains("rep.1"), "{}", stdout);
    assert!(stdout.contains("Wald Tests"), "{}", stdout);
}

#[test]
fn fit_json_output_and_blups_csv() {
    let data = temp_file("trial.csv", TRIAL_CSV);
    let blups = std::env::temp_dir().join(format!("openblup_blups_{}.csv", std::process::id()));
    let out = openblup()
        .args([
            "fit",
            "--data",
            data.to_str().unwrap(),
            "--response",
            "yield",
            "--fixed",
            "rep",
            "--factor",
            "rep",
            "--random",
            "genotype",
            "--algorithm",
            "em",
            "--format",
            "json",
            "--blups",
            blups.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["n_obs"], 6);
    assert_eq!(json["random_effects"][0]["term"], "genotype");
    assert_eq!(json["random_effects"][0]["n_levels"], 3);
    assert_eq!(
        json["random_effects"][0]["effects"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert!(json["variance_components"].as_array().unwrap().len() == 2);
    // BLUP ranking G1 > G2 > G3
    let effects = json["random_effects"][0]["effects"].as_array().unwrap();
    let est = |lvl: &str| {
        effects.iter().find(|e| e["level"] == lvl).unwrap()["estimate"]
            .as_f64()
            .unwrap()
    };
    assert!(est("G1") > est("G2") && est("G2") > est("G3"));

    let csv = std::fs::read_to_string(&blups).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines[0], "term,level,estimate,se,reliability");
    assert_eq!(lines.len(), 4);
    assert!(lines[1].starts_with("genotype,G"));
    std::fs::remove_file(&blups).ok();
}

#[test]
fn fit_with_pedigree_reproduces_mrode_ranking() {
    let data = temp_file("mrode.csv", MRODE_DATA);
    let ped = temp_file("mrode_ped.csv", MRODE_PED);
    let out = openblup()
        .args([
            "fit",
            "--data",
            data.to_str().unwrap(),
            "--response",
            "pwg",
            "--fixed",
            "sex",
            "--random",
            "animal",
            "--pedigree",
            ped.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {}", stderr);
    assert!(
        stderr.contains("Using pedigree A-inverse (8x8"),
        "{}",
        stderr
    );
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    // All 8 pedigree animals get a breeding value, not just the 5 with records.
    assert_eq!(json["random_effects"][0]["n_levels"], 8);
    let effects = json["random_effects"][0]["effects"].as_array().unwrap();
    let est = |lvl: &str| {
        effects.iter().find(|e| e["level"] == lvl).unwrap()["estimate"]
            .as_f64()
            .unwrap()
    };
    // Mrode's ranking at the known variance ratio: 8 > 6 > 1 > 2 > 4 > 3 > 5 > 7.
    // REML estimates the ratio from 5 records, so we only check the extremes.
    assert!(est("8") > est("7"));
    assert!(est("6") > est("5"));
    // Fixed effects: male > female
    let fe = json["fixed_effects"].as_array().unwrap();
    let male = fe.iter().find(|e| e["level"] == "male").unwrap()["estimate"]
        .as_f64()
        .unwrap();
    let female = fe.iter().find(|e| e["level"] == "female").unwrap()["estimate"]
        .as_f64()
        .unwrap();
    assert!(male > female);
}

#[test]
fn fit_pedigree_without_random_term_is_an_error() {
    let data = temp_file("mrode.csv", MRODE_DATA);
    let ped = temp_file("mrode_ped.csv", MRODE_PED);
    let out = openblup()
        .args([
            "fit",
            "--data",
            data.to_str().unwrap(),
            "--response",
            "pwg",
            "--pedigree",
            ped.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--random"));
}

#[test]
fn ainverse_with_inbreeding_and_output() {
    let ped = temp_file("mrode_ped.csv", MRODE_PED);
    let outfile = std::env::temp_dir().join(format!("openblup_ainv_{}.csv", std::process::id()));
    let out = openblup()
        .args([
            "ainverse",
            "--pedigree",
            ped.to_str().unwrap(),
            "--inbreeding",
            "--output",
            outfile.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("A-inverse dimensions: 8 x 8"), "{}", stdout);
    assert!(stdout.contains("Meuwissen & Luo"), "{}", stdout);
    assert!(stdout.contains("0 of 8 animals inbred"), "{}", stdout);

    let csv = std::fs::read_to_string(&outfile).unwrap();
    assert!(csv.starts_with("row,col,value\n"));
    // Animal 1 (base, sire of 4 and 6): diagonal = 1 + 4/3/4 + 2/4 = 1.8333
    let diag_1 = csv
        .lines()
        .find(|l| l.starts_with("1,1,"))
        .expect("diagonal entry for animal 1");
    let val: f64 = diag_1.split(',').nth(2).unwrap().parse().unwrap();
    assert!((val - 1.8333).abs() < 1e-3, "{}", val);
    std::fs::remove_file(&outfile).ok();
}

#[test]
fn version_flag_works() {
    let out = openblup().arg("--version").output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("openblup"));
}
