#[derive(Debug, Clone, PartialEq, Default)]
pub struct MapEntry {
    pub vma: u64,
    pub lma: u64,
    pub size: u64,
    pub align: u64,
    pub section_out: Option<String>,
    pub section_in: Option<String>,
    pub symbol: Option<String>,
}

pub struct MapParser;

impl MapParser {
    pub fn parse(input: &str) -> Vec<MapEntry> {
        if input.contains("VMA") && input.contains("LMA") && input.contains("Align") {
            Self::parse_lld(input)
        } else {
            Self::parse_gnu(input)
        }
    }

    /// Parses LLVM LLD map files (Tabular format)
    fn parse_lld(input: &str) -> Vec<MapEntry> {
        let mut entries = Vec::new();
        for line in input.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 5 || parts[0] == "VMA" {
                continue;
            }

            let vma = u64::from_str_radix(parts[0], 16).unwrap_or(0);
            let lma = u64::from_str_radix(parts[1], 16).unwrap_or(0);
            let size = u64::from_str_radix(parts[2], 16).unwrap_or(0);
            let align = parts[3].parse::<u64>().unwrap_or(0);
            let label = parts[4..].join(" ");

            let mut entry = MapEntry {
                vma,
                lma,
                size,
                align,
                ..Default::default()
            };

            // LLD logic: Indentation-based categories
            if line.starts_with(' ') {
                if label.contains(':') {
                    entry.section_in = Some(label);
                } else {
                    entry.symbol = Some(label);
                }
            } else {
                entry.section_out = Some(label);
            }
            entries.push(entry);
        }
        entries
    }

    /// Parses GNU ld map files (Multi-line/Complex format)
    fn parse_gnu(input: &str) -> Vec<MapEntry> {
        let mut entries = Vec::new();
        let mut lines = input
            .lines()
            .skip_while(|l| !l.contains("Linker script and memory map"));

        // Skip the header line itself
        lines.next();

        let mut current_section_out: Option<String> = None;
        let mut buffered_line: Option<String> = None;

        while let Some(line) = lines.next().or(buffered_line.take().as_deref()) {
            if line.trim().is_empty() || line.starts_with("LOAD") || line.starts_with("OUTPUT") {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();

            // Handle GNU line wrapping:
            // Often: ".text\n  0x0000000000401000 0xb0"
            if parts.len() == 1 && !parts[0].starts_with("0x") {
                let next_line = lines.next().unwrap_or("");
                let combined = format!("{} {}", line, next_line);
                let merged_parts: Vec<&str> = combined.split_whitespace().collect();
                if merged_parts.len() >= 3 {
                    if let Some(entry) =
                        Self::parse_gnu_line(&merged_parts, &mut current_section_out)
                    {
                        entries.push(entry);
                    }
                }
                continue;
            }

            if let Some(entry) = Self::parse_gnu_line(&parts, &mut current_section_out) {
                entries.push(entry);
            }
        }
        entries
    }

    fn parse_gnu_line(parts: &[&str], current_out: &mut Option<String>) -> Option<MapEntry> {
        // GNU format: [Name] [Address] [Size] [File/Symbol]
        // Address and size in GNU usually start with 0x
        let mut addr_idx = None;
        for (i, part) in parts.iter().enumerate() {
            if part.starts_with("0x") {
                addr_idx = Some(i);
                break;
            }
        }

        let idx = addr_idx?;
        let vma = u64::from_str_radix(parts[idx].trim_start_matches("0x"), 16).ok()?;
        let size = if parts.len() > idx + 1 {
            u64::from_str_radix(parts[idx + 1].trim_start_matches("0x"), 16).unwrap_or(0)
        } else {
            0
        };

        let mut entry = MapEntry {
            vma,
            lma: vma, // GNU LD map files usually show VMA; LMA is hidden unless using AT
            size,
            align: 0, // GNU map doesn't provide alignment per line easily
            ..Default::default()
        };

        if idx == 1 {
            // .text 0x00401000 0xb0
            let name = parts[0].to_string();
            if name.starts_with('.') {
                *current_out = Some(name.clone());
                entry.section_out = Some(name);
            } else {
                entry.section_in = Some(name);
            }
        } else if idx == 0 {
            // 0x004010a0 symbol_name
            entry.symbol = parts.get(1).map(|s| s.to_string());
            entry.section_out = current_out.clone();
        }

        Some(entry)
    }
}
