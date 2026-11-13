//! An ad-hoc code signature over an image's code, in the form Darwin's kernel takes.
//!
//! macOS maps a file's pages executable only when a code signature covers them, and
//! `fcntl(F_ADDSIGS)` attaches one to any file, not only a Mach-O. The signature is a
//! superblob holding one code directory: the SHA-256 of each page and no identity.

use sha2::{Digest, Sha256};

/// log2 of the page size each hash covers.
pub const PAGE_SHIFT: u8 = 14;
pub const PAGE: usize = 1 << PAGE_SHIFT;

const SUPERBLOB_MAGIC: u32 = 0xfade_0cc0;
const CODE_DIRECTORY_MAGIC: u32 = 0xfade_0c02;
const CODE_DIRECTORY_SLOT: u32 = 0;
/// The code directory version with 64-bit limits and the executable segment fields.
const CODE_DIRECTORY_VERSION: u32 = 0x2_0400;
const CS_ADHOC: u32 = 0x2;
const HASH_SHA256: u8 = 2;
const HASH_SIZE: usize = 32;
/// Every field through `execSegFlags`.
const CODE_DIRECTORY_HEADER: usize = 88;
const IDENTIFIER: &[u8] = b"microlift.image\0";

/// Hashes code as it is written, a page at a time.
#[derive(Default)]
pub struct PageHasher {
    page: Sha256,
    filled: usize,
    hashes: Vec<[u8; HASH_SIZE]>,
    len: u64,
}

impl PageHasher {
    pub fn update(&mut self, mut bytes: &[u8]) {
        self.len += bytes.len() as u64;
        while !bytes.is_empty() {
            let take = (PAGE - self.filled).min(bytes.len());
            self.page.update(&bytes[..take]);
            self.filled += take;
            bytes = &bytes[take..];
            if self.filled == PAGE {
                self.hashes
                    .push(std::mem::take(&mut self.page).finalize().into());
                self.filled = 0;
            }
        }
    }

    /// The superblob signing everything hashed, mapped from its first byte.
    pub fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            self.hashes.push(self.page.finalize().into());
        }
        superblob(&self.hashes, self.len)
    }
}

fn superblob(hashes: &[[u8; HASH_SIZE]], code_len: u64) -> Vec<u8> {
    let ident_offset = CODE_DIRECTORY_HEADER;
    let hash_offset = ident_offset + IDENTIFIER.len();
    let directory_len = hash_offset + hashes.len() * HASH_SIZE;
    let mut directory = Vec::with_capacity(directory_len);
    let be32 = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_be_bytes());
    let be64 = |out: &mut Vec<u8>, v: u64| out.extend_from_slice(&v.to_be_bytes());
    be32(&mut directory, CODE_DIRECTORY_MAGIC);
    be32(&mut directory, directory_len as u32);
    be32(&mut directory, CODE_DIRECTORY_VERSION);
    be32(&mut directory, CS_ADHOC);
    be32(&mut directory, hash_offset as u32);
    be32(&mut directory, ident_offset as u32);
    // No special slots.
    be32(&mut directory, 0);
    be32(&mut directory, hashes.len() as u32);
    // The 32-bit limit is zero when only the 64-bit one can hold the length.
    let limit32 = u32::try_from(code_len).unwrap_or(0);
    be32(&mut directory, limit32);
    directory.extend_from_slice(&[HASH_SIZE as u8, HASH_SHA256, 0, PAGE_SHIFT]);
    // spare2, scatterOffset, teamOffset, spare3
    for _ in 0..4 {
        be32(&mut directory, 0);
    }
    be64(&mut directory, if limit32 == 0 { code_len } else { 0 });
    // execSegBase, execSegLimit, execSegFlags
    be64(&mut directory, 0);
    be64(&mut directory, code_len);
    be64(&mut directory, 0);
    debug_assert_eq!(directory.len(), CODE_DIRECTORY_HEADER);
    directory.extend_from_slice(IDENTIFIER);
    for hash in hashes {
        directory.extend_from_slice(hash);
    }

    let index_len = 12 + 8;
    let mut blob = Vec::with_capacity(index_len + directory.len());
    be32(&mut blob, SUPERBLOB_MAGIC);
    be32(&mut blob, (index_len + directory.len()) as u32);
    be32(&mut blob, 1);
    be32(&mut blob, CODE_DIRECTORY_SLOT);
    be32(&mut blob, index_len as u32);
    blob.extend_from_slice(&directory);
    blob
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_signature_hashes_every_page_including_a_partial_last_one() {
        let code = vec![0xabu8; PAGE + 10];
        let mut hasher = PageHasher::default();
        hasher.update(&code[..100]);
        hasher.update(&code[100..]);
        let blob = hasher.finish();
        let directory = &blob[20..];
        let at = |i: usize| u32::from_be_bytes(directory[i..i + 4].try_into().unwrap());
        assert_eq!(at(0), CODE_DIRECTORY_MAGIC);
        assert_eq!(at(28), 2, "two code slots");
        let hash_offset = at(16) as usize;
        let second: [u8; 32] = Sha256::digest(&code[PAGE..]).into();
        assert_eq!(&directory[hash_offset + 32..hash_offset + 64], &second);
    }
}
