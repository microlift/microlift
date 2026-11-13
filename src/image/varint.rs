//! LEB128 integers, with zigzag for signed ones: most of what an image records per
//! region is a small distance from the region's own address.

pub(crate) fn put(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

pub(crate) fn put_signed(out: &mut Vec<u8>, value: i64) {
    put(out, ((value << 1) ^ (value >> 63)) as u64);
}

/// Reads integers back from a byte slice, failing rather than reading past it.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    pub(crate) fn get(&mut self) -> crate::Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self
                .bytes
                .get(self.at)
                .ok_or_else(|| super::malformed("truncated integer"))?;
            self.at += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(super::malformed("integer too long"))
    }

    pub(crate) fn get_signed(&mut self) -> crate::Result<i64> {
        let value = self.get()?;
        Ok((value >> 1) as i64 ^ -((value & 1) as i64))
    }

    pub(crate) fn get_u32(&mut self) -> crate::Result<u32> {
        u32::try_from(self.get()?).map_err(|_| super::malformed("integer too large"))
    }

    /// A count of items each at least one byte long, bounded by what is left.
    pub(crate) fn count(&mut self) -> crate::Result<usize> {
        usize::try_from(self.get()?)
            .ok()
            .filter(|&n| n <= self.bytes.len() - self.at)
            .ok_or_else(|| super::malformed("count exceeds the record"))
    }

    pub(crate) fn bytes(&mut self) -> crate::Result<&'a [u8]> {
        let n = self.count()?;
        let slice = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(slice)
    }

    pub(crate) fn str(&mut self) -> crate::Result<&'a str> {
        std::str::from_utf8(self.bytes()?).map_err(|_| super::malformed("string is not UTF-8"))
    }

    pub(crate) fn byte(&mut self) -> crate::Result<u8> {
        let byte = *self
            .bytes
            .get(self.at)
            .ok_or_else(|| super::malformed("truncated record"))?;
        self.at += 1;
        Ok(byte)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.at == self.bytes.len()
    }
}

pub(crate) fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_survive_encoding() {
        let mut out = Vec::new();
        let values = [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX];
        let signed = [0i64, -1, 1, -64, 64, i64::MIN, i64::MAX];
        values.iter().for_each(|&v| put(&mut out, v));
        signed.iter().for_each(|&v| put_signed(&mut out, v));
        let mut r = Reader::new(&out);
        for &v in &values {
            assert_eq!(r.get().unwrap(), v);
        }
        for &v in &signed {
            assert_eq!(r.get_signed().unwrap(), v);
        }
        assert!(r.is_empty());
        assert!(r.get().is_err());
    }
}
