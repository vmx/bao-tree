//! Async io, written in fsm style
//!
//! IO ops are written as async state machines that thread the state through the
//! futures to avoid being encumbered by lifetimes.
//!
//! This makes them occasionally a bit verbose to use, but allows being generic
//! without having to box the futures.
//!
//! The traits to perform async io are re-exported from
//! [iroh-io](https://crates.io/crates/iroh-io).
use std::{
    future::Future,
    io::{self, Cursor},
    result,
};

use bytes::Bytes;
pub use iroh_io::{AsyncSliceReader, AsyncSliceWriter};
use iroh_io::{AsyncStreamReader, AsyncStreamWriter};
use smallvec::SmallVec;

pub use super::BaoContentItem;
use super::{combine_hash_pair, DecodeError};
use crate::{
    io::{
        error::EncodeError,
        outboard::{PostOrderOutboard, PreOrderOutboard},
        Leaf, Parent,
    },
    iter::{BaoChunk, ResponseIter},
    rec::{encode_selected_rec, truncate_ranges, truncate_ranges_owned},
    BaoTree, BlockSize, ChunkRanges, ChunkRangesRef, Hash, Hasher, TreeNode,
};

/// A binary merkle tree for blake3 hashes of a blob.
///
/// This trait contains information about the geometry of the tree, the root hash,
/// and a method to load the hashes at a given node.
///
/// It is up to the implementor to decide how to store the hashes.
///
/// In the original bao crate, the hashes are stored in a file in pre order.
/// This is implemented for a generic io object in [super::outboard::PreOrderOutboard]
/// and for a memory region in [super::outboard::PreOrderMemOutboard].
///
/// For files that grow over time, it is more efficient to store the hashes in post order.
/// This is implemented for a generic io object in [super::outboard::PostOrderOutboard]
/// and for a memory region in [super::outboard::PostOrderMemOutboard].
///
/// If you use a different storage engine, you can implement this trait for it. E.g.
/// you could store the hashes in a database and use the node number as the key.
///
/// The async version takes a mutable reference to load, not because it mutates
/// the outboard (it doesn't), but to ensure that there is at most one outstanding
/// load at a time.
///
/// Dropping the load future without polling it to completion is safe, but will
/// possibly render the outboard unusable.
pub trait Outboard {
    /// The hasher that is used
    type Hasher: Hasher;

    /// The root hash
    fn root(&self) -> Hash;
    /// The tree. This contains the information about the size of the file and the block size.
    fn tree(&self) -> BaoTree;
    /// load the hash pair for a node
    ///
    /// This takes a &mut self not because it mutates the outboard (it doesn't),
    /// but to ensure that there is only one outstanding load at a time.
    fn load(&mut self, node: TreeNode) -> impl Future<Output = io::Result<Option<(Hash, Hash)>>>;
}

/// A mutable outboard.
///
/// This trait provides a way to save a hash pair for a node and to set the
/// length of the data file.
///
/// This trait can be used to incrementally save an outboard when receiving data.
/// If you want to just ignore outboard data, there is a special placeholder outboard
/// implementation [super::outboard::EmptyOutboard].
pub trait OutboardMut: Sized {
    /// The hasher that is used
    type Hasher: Hasher;

    /// Save a hash pair for a node
    fn save(
        &mut self,
        node: TreeNode,
        hash_pair: &(Hash, Hash),
    ) -> impl Future<Output = io::Result<()>>;

    /// sync to disk
    fn sync(&mut self) -> impl Future<Output = io::Result<()>>;
}

/// Convenience trait to initialize an outboard from a data source.
///
/// In complex real applications, you might want to do this manually.
pub trait CreateOutboard {
    /// Create an outboard from a seekable data source, measuring the size first.
    ///
    /// This requires the outboard to have a default implementation, which is
    /// the case for the memory implementations.
    #[allow(async_fn_in_trait)]
    async fn create(mut data: impl AsyncSliceReader, block_size: BlockSize) -> io::Result<Self>
    where
        Self: Default + Sized,
    {
        let size = data.size().await?;
        Self::create_sized(Cursor::new(data), size, block_size).await
    }

    /// create an outboard from a data source. This requires the outboard to
    /// have a default implementation, which is the case for the memory
    /// implementations.
    fn create_sized(
        data: impl AsyncStreamReader,
        size: u64,
        block_size: BlockSize,
    ) -> impl Future<Output = io::Result<Self>>
    where
        Self: Default + Sized;

    /// Init the outboard from a data source. This will use the existing
    /// tree and only init the data and set the root hash.
    ///
    /// So this can be used to initialize an outboard that does not have a default,
    /// such as a file based one.
    ///
    /// It will only include data up the the current tree size.
    fn init_from(&mut self, data: impl AsyncStreamReader) -> impl Future<Output = io::Result<()>>;
}

impl<O: Outboard> Outboard for &mut O {
    type Hasher = O::Hasher;

    fn root(&self) -> Hash {
        (**self).root()
    }

    fn tree(&self) -> BaoTree {
        (**self).tree()
    }

    async fn load(&mut self, node: TreeNode) -> io::Result<Option<(Hash, Hash)>> {
        (**self).load(node).await
    }
}

impl<R: AsyncSliceReader, H: Hasher> Outboard for PreOrderOutboard<R, H> {
    type Hasher = H;

    fn root(&self) -> Hash {
        self.root.clone()
    }

    fn tree(&self) -> BaoTree {
        self.tree
    }

    async fn load(&mut self, node: TreeNode) -> io::Result<Option<(Hash, Hash)>> {
        let Some(offset) = self.tree.pre_order_offset(node) else {
            return Ok(None);
        };
        let offset = offset * 64;
        let content = self.data.read_at(offset, 64).await?;
        Ok(Some(if content.len() != 64 {
            (Hash::from([0; 32]), Hash::from([0; 32]))
        } else {
            parse_hash_pair(content)?
        }))
    }
}

impl<O: OutboardMut> OutboardMut for &mut O {
    type Hasher = O::Hasher;

    async fn save(&mut self, node: TreeNode, hash_pair: &(Hash, Hash)) -> io::Result<()> {
        (**self).save(node, hash_pair).await
    }

    async fn sync(&mut self) -> io::Result<()> {
        (**self).sync().await
    }
}

impl<W: AsyncSliceWriter, H: Hasher> OutboardMut for PreOrderOutboard<W, H> {
    type Hasher = H;

    async fn save(&mut self, node: TreeNode, hash_pair: &(Hash, Hash)) -> io::Result<()> {
        let Some(offset) = self.tree.pre_order_offset(node) else {
            return Ok(());
        };
        let offset = offset * 64;
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(hash_pair.0.as_bytes());
        buf[32..].copy_from_slice(hash_pair.1.as_bytes());
        self.data.write_at(offset, &buf).await?;
        Ok(())
    }

    async fn sync(&mut self) -> io::Result<()> {
        self.data.sync().await
    }
}

impl<W: AsyncSliceWriter, H: Hasher> OutboardMut for PostOrderOutboard<W, H> {
    type Hasher = H;

    async fn save(&mut self, node: TreeNode, hash_pair: &(Hash, Hash)) -> io::Result<()> {
        let Some(offset) = self.tree.post_order_offset(node) else {
            return Ok(());
        };
        let offset = offset.value() * 64;
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(hash_pair.0.as_bytes());
        buf[32..].copy_from_slice(hash_pair.1.as_bytes());
        self.data.write_at(offset, &buf).await?;
        Ok(())
    }

    async fn sync(&mut self) -> io::Result<()> {
        self.data.sync().await
    }
}

impl<W: AsyncSliceWriter, H: Hasher> CreateOutboard for PreOrderOutboard<W, H> {
    async fn create_sized(
        data: impl AsyncStreamReader,
        size: u64,
        block_size: BlockSize,
    ) -> io::Result<Self>
    where
        Self: Default + Sized,
    {
        let mut res = Self {
            tree: BaoTree::new(size, block_size),
            ..Self::default()
        };
        res.init_from(data).await?;
        Ok(res)
    }

    async fn init_from(&mut self, data: impl AsyncStreamReader) -> io::Result<()> {
        let mut this = self;
        let root = outboard(data, this.tree, &mut this).await?;
        this.root = root;
        this.sync().await?;
        Ok(())
    }
}

impl<W: AsyncSliceWriter, H: Hasher> CreateOutboard for PostOrderOutboard<W, H> {
    async fn create_sized(
        data: impl AsyncStreamReader,
        size: u64,
        block_size: BlockSize,
    ) -> io::Result<Self>
    where
        Self: Default + Sized,
    {
        let mut res = Self {
            tree: BaoTree::new(size, block_size),
            ..Self::default()
        };
        res.init_from(data).await?;
        Ok(res)
    }

    async fn init_from(&mut self, data: impl AsyncStreamReader) -> io::Result<()> {
        let mut this = self;
        let root = outboard(data, this.tree, &mut this).await?;
        this.root = root;
        this.sync().await?;
        Ok(())
    }
}

impl<R: AsyncSliceReader, H: Hasher> Outboard for PostOrderOutboard<R, H> {
    type Hasher = H;

    fn root(&self) -> Hash {
        self.root.clone()
    }

    fn tree(&self) -> BaoTree {
        self.tree
    }

    async fn load(&mut self, node: TreeNode) -> io::Result<Option<(Hash, Hash)>> {
        let Some(offset) = self.tree.post_order_offset(node) else {
            return Ok(None);
        };
        let offset = offset.value() * 64;
        let content = self.data.read_at(offset, 64).await?;
        Ok(Some(if content.len() != 64 {
            (Hash::from([0; 32]), Hash::from([0; 32]))
        } else {
            parse_hash_pair(content)?
        }))
    }
}

pub(crate) fn parse_hash_pair(buf: Bytes) -> io::Result<(Hash, Hash)> {
    if buf.len() != 64 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "hash pair must be 64 bytes",
        ));
    }
    let l_hash = Hash::from(<[u8; 32]>::try_from(&buf[..32]).unwrap());
    let r_hash = Hash::from(<[u8; 32]>::try_from(&buf[32..]).unwrap());
    Ok((l_hash, r_hash))
}

#[derive(Debug)]
struct ResponseDecoderInner<R> {
    iter: ResponseIter,
    stack: SmallVec<[Hash; 10]>,
    encoded: R,
}

impl<R> ResponseDecoderInner<R> {
    fn new(tree: BaoTree, hash: Hash, ranges: ChunkRanges, encoded: R) -> Self {
        // now that we know the size, we can canonicalize the ranges
        let ranges = truncate_ranges_owned(ranges, tree.size());
        let mut res = Self {
            iter: ResponseIter::new(tree, ranges),
            stack: SmallVec::new(),
            encoded,
        };
        res.stack.push(hash);
        res
    }
}

/// Response decoder
#[derive(Debug)]
pub struct ResponseDecoder<R, H> {
    inner: Box<ResponseDecoderInner<R>>,
    hasher: std::marker::PhantomData<H>,
}

/// Next type for ResponseDecoder.
#[derive(Debug)]
pub enum ResponseDecoderNext<R, H> {
    /// One more item, and you get back the state machine in the next state
    More(
        (
            ResponseDecoder<R, H>,
            std::result::Result<BaoContentItem, DecodeError>,
        ),
    ),
    /// The stream is done, you get back the underlying reader
    Done(R),
}

impl<R: AsyncStreamReader, H: Hasher> ResponseDecoder<R, H> {
    /// Create a new response decoder state machine, when you have already read the size.
    ///
    /// The size as well as the chunk size is given in the `tree` parameter.
    pub fn new(hash: Hash, ranges: ChunkRanges, tree: BaoTree, encoded: R) -> Self {
        Self {
            inner: Box::new(ResponseDecoderInner::new(tree, hash, ranges, encoded)),
            hasher: std::marker::PhantomData::<H>,
        }
    }

    /// Proceed to the next state by reading the next chunk from the stream.
    pub async fn next(mut self) -> ResponseDecoderNext<R, H> {
        if let Some(chunk) = self.inner.iter.next() {
            let item = self.next0(chunk).await;
            ResponseDecoderNext::More((self, item))
        } else {
            ResponseDecoderNext::Done(self.inner.encoded)
        }
    }

    /// Immediately return the underlying reader
    pub fn finish(self) -> R {
        self.inner.encoded
    }

    /// The tree geometry
    pub fn tree(&self) -> BaoTree {
        self.inner.iter.tree()
    }

    /// Hash of the blob we are currently getting
    pub fn hash(&self) -> &Hash {
        &self.inner.stack[0]
    }

    async fn next0(&mut self, chunk: BaoChunk) -> std::result::Result<BaoContentItem, DecodeError> {
        Ok(match chunk {
            BaoChunk::Parent {
                is_root,
                right,
                left,
                node,
                ..
            } => {
                let this = &mut self.inner;
                let buf = this
                    .encoded
                    .read::<64>()
                    .await
                    .map_err(|e| DecodeError::maybe_parent_not_found(e, node))?;
                let ref pair @ (ref l_hash, ref r_hash) = read_parent(&buf);
                let parent_hash = this.stack.pop().unwrap();
                let actual = H::hash_inner(&l_hash, &r_hash, is_root);
                // Push the children in reverse order so they are popped in the correct order
                // only push right if the range intersects with the right child
                if right {
                    this.stack.push(r_hash.clone());
                }
                // only push left if the range intersects with the left child
                if left {
                    this.stack.push(l_hash.clone());
                }
                // Validate after pushing the children so that we could in principle continue
                if parent_hash != actual {
                    return Err(DecodeError::ParentHashMismatch(node));
                }
                Parent {
                    pair: pair.clone(),
                    node,
                }
                .into()
            }
            BaoChunk::Leaf {
                size,
                is_root,
                start_chunk,
                ..
            } => {
                // this will resize always to chunk group size, except for the last chunk
                let this = &mut self.inner;
                let data = this
                    .encoded
                    .read_bytes_exact(size)
                    .await
                    .map_err(|e| DecodeError::maybe_leaf_not_found(e, start_chunk))?;
                let leaf_hash = this.stack.pop().unwrap();
                let actual = H::hash_chunk(start_chunk.0, &data, is_root);
                if leaf_hash != actual {
                    return Err(DecodeError::LeafHashMismatch(start_chunk));
                }
                Leaf {
                    offset: start_chunk.to_bytes(),
                    data,
                }
                .into()
            }
        })
    }
}

/// Encode ranges relevant to a query from a reader and outboard to a writer
///
/// This will not validate on writing, so data corruption will be detected on reading
///
/// It is possible to encode ranges from a partial file and outboard.
/// This will either succeed if the requested ranges are all present, or fail
/// as soon as a range is missing.
pub async fn encode_ranges<D, O, W>(
    mut data: D,
    mut outboard: O,
    ranges: &ChunkRangesRef,
    encoded: W,
) -> result::Result<(), EncodeError>
where
    D: AsyncSliceReader,
    O: Outboard,
    W: AsyncStreamWriter,
{
    let mut encoded = encoded;
    let tree = outboard.tree();
    for item in tree.ranges_pre_order_chunks_iter_ref(ranges, 0) {
        match item {
            BaoChunk::Parent { node, .. } => {
                let (l_hash, r_hash) = outboard.load(node).await?.unwrap();
                let pair = combine_hash_pair(&l_hash, &r_hash);
                encoded
                    .write(&pair)
                    .await
                    .map_err(|e| EncodeError::maybe_parent_write(e, node))?;
            }
            BaoChunk::Leaf {
                start_chunk, size, ..
            } => {
                let start = start_chunk.to_bytes();
                let bytes = data.read_exact_at(start, size).await?;
                encoded
                    .write_bytes(bytes)
                    .await
                    .map_err(|e| EncodeError::maybe_leaf_write(e, start_chunk))?;
            }
        }
    }
    Ok(())
}

/// Encode ranges relevant to a query from a reader and outboard to a writer
///
/// This function validates the data before writing
///
/// It is possible to encode ranges from a partial file and outboard.
/// This will either succeed if the requested ranges are all present, or fail
/// as soon as a range is missing.
pub async fn encode_ranges_validated<D, O, W>(
    mut data: D,
    mut outboard: O,
    ranges: &ChunkRangesRef,
    encoded: W,
) -> result::Result<(), EncodeError>
where
    D: AsyncSliceReader,
    O: Outboard,
    W: AsyncStreamWriter,
{
    // buffer for writing incomplete subtrees.
    // for queries that don't have incomplete subtrees, this will never be used.
    let mut out_buf = Vec::new();
    let mut stack = SmallVec::<[Hash; 10]>::new();
    stack.push(outboard.root());
    let mut encoded = encoded;
    let tree = outboard.tree();
    let ranges = truncate_ranges(ranges, tree.size());
    for item in tree.ranges_pre_order_chunks_iter_ref(ranges, 0) {
        match item {
            BaoChunk::Parent {
                is_root,
                left,
                right,
                node,
                ..
            } => {
                let (l_hash, r_hash) = outboard.load(node).await?.unwrap();
                let actual = O::Hasher::hash_inner(&l_hash, &r_hash, is_root);
                let expected = stack.pop().unwrap();
                if actual != expected {
                    return Err(EncodeError::ParentHashMismatch(node));
                }
                if right {
                    stack.push(r_hash.clone());
                }
                if left {
                    stack.push(l_hash.clone());
                }
                let pair = combine_hash_pair(&l_hash, &r_hash);
                encoded
                    .write(&pair)
                    .await
                    .map_err(|e| EncodeError::maybe_parent_write(e, node))?;
            }
            BaoChunk::Leaf {
                start_chunk,
                size,
                is_root,
                ranges,
                ..
            } => {
                let expected = stack.pop().unwrap();
                let start = start_chunk.to_bytes();
                let bytes = data.read_exact_at(start, size).await?;
                let (actual, to_write) = if !ranges.is_all() {
                    // we need to encode just a part of the data
                    //
                    // write into an out buffer to ensure we detect mismatches
                    // before writing to the output.
                    out_buf.clear();
                    let actual = encode_selected_rec::<O::Hasher>(
                        start_chunk,
                        &bytes,
                        is_root,
                        ranges,
                        tree.block_size.to_u32(),
                        true,
                        &mut out_buf,
                    );
                    (actual, out_buf.clone().into())
                } else {
                    let actual = O::Hasher::hash_chunk(start_chunk.0, &bytes, is_root);
                    (actual, bytes)
                };
                if actual != expected {
                    return Err(EncodeError::LeafHashMismatch(start_chunk));
                }
                encoded
                    .write_bytes(to_write)
                    .await
                    .map_err(|e| EncodeError::maybe_leaf_write(e, start_chunk))?;
            }
        }
    }
    Ok(())
}

/// Decode a response into a file while updating an outboard.
///
/// If you do not want to update an outboard, use [super::outboard::EmptyOutboard] as
/// the outboard.
pub async fn decode_ranges<R, O, W>(
    encoded: R,
    ranges: ChunkRanges,
    mut target: W,
    mut outboard: O,
) -> std::result::Result<(), DecodeError>
where
    O: OutboardMut + Outboard,
    R: AsyncStreamReader,
    W: AsyncSliceWriter,
{
    let mut reading = ResponseDecoder::<R, <O as Outboard>::Hasher>::new(
        outboard.root(),
        ranges,
        outboard.tree(),
        encoded,
    );
    loop {
        let item = match reading.next().await {
            ResponseDecoderNext::Done(_reader) => break,
            ResponseDecoderNext::More((next, item)) => {
                reading = next;
                item?
            }
        };
        match item {
            BaoContentItem::Parent(Parent { node, pair }) => {
                outboard.save(node, &pair).await?;
            }
            BaoContentItem::Leaf(Leaf { offset, data }) => {
                target.write_bytes_at(offset, data).await?;
            }
        }
    }
    Ok(())
}
fn read_parent(buf: &[u8]) -> (Hash, Hash) {
    let l_hash = Hash::from(<[u8; 32]>::try_from(&buf[..32]).unwrap());
    let r_hash = Hash::from(<[u8; 32]>::try_from(&buf[32..64]).unwrap());
    (l_hash, r_hash)
}

/// Compute the outboard for the given data.
///
/// Unlike [outboard_post_order], this will work with any outboard
/// implementation, but it is not guaranteed that writes are sequential.
pub async fn outboard<R: AsyncStreamReader, O: OutboardMut>(
    mut data: R,
    tree: BaoTree,
    mut outboard: O,
) -> io::Result<Hash> {
    // do not allocate for small trees
    let mut stack = SmallVec::<[Hash; 10]>::new();
    for item in tree.post_order_chunks_iter() {
        match item {
            BaoChunk::Parent { is_root, node, .. } => {
                let right_hash = stack.pop().unwrap();
                let left_hash = stack.pop().unwrap();
                outboard
                    .save(node, &(left_hash.clone(), right_hash.clone()))
                    .await?;
                let parent = O::Hasher::hash_inner(&left_hash, &right_hash, is_root);
                stack.push(parent);
            }
            BaoChunk::Leaf {
                size,
                is_root,
                start_chunk,
                ..
            } => {
                let buf = data.read_bytes_exact(size).await?;
                let hash = O::Hasher::hash_chunk(start_chunk.0, &buf, is_root);
                stack.push(hash);
            }
        }
    }
    debug_assert_eq!(stack.len(), 1);
    let hash = stack.pop().unwrap();
    Ok(hash)
}

/// Compute the post order outboard for the given data, writing into a io::Write
///
/// For the post order outboard, writes to the target are sequential.
///
/// This will not add the size to the output. You need to store it somewhere else
/// or append it yourself.
pub async fn outboard_post_order<H: Hasher>(
    mut data: impl AsyncStreamReader,
    tree: BaoTree,
    mut outboard: impl AsyncStreamWriter,
) -> io::Result<Hash> {
    // do not allocate for small trees
    let mut stack = SmallVec::<[Hash; 10]>::new();
    for item in tree.post_order_chunks_iter() {
        match item {
            BaoChunk::Parent { is_root, .. } => {
                let right_hash = stack.pop().unwrap();
                let left_hash = stack.pop().unwrap();
                outboard.write(&left_hash.0).await?;
                outboard.write(&right_hash.0).await?;
                let parent = H::hash_inner(&left_hash, &right_hash, is_root);
                stack.push(parent);
            }
            BaoChunk::Leaf {
                size,
                is_root,
                start_chunk,
                ..
            } => {
                let buf = data.read_bytes_exact(size).await?;
                let hash = H::hash_chunk(start_chunk.0, &buf, is_root);
                stack.push(hash);
            }
        }
    }
    debug_assert_eq!(stack.len(), 1);
    let hash = stack.pop().unwrap();
    Ok(hash)
}

/// Copy an outboard to another outboard.
///
/// This can be used to persist an in memory outboard or to change from
/// pre-order to post-order.
pub async fn copy(mut from: impl Outboard, mut to: impl OutboardMut) -> io::Result<()> {
    let tree = from.tree();
    for node in tree.pre_order_nodes_iter() {
        if let Some(hash_pair) = from.load(node).await? {
            to.save(node, &hash_pair).await?;
        }
    }
    Ok(())
}

#[cfg(feature = "validate")]
mod validate {
    use std::{io, ops::Range};

    use futures_lite::{FutureExt, Stream};
    use genawaiter::sync::{Co, Gen};
    use iroh_io::AsyncSliceReader;

    use super::Outboard;
    use crate::{
        io::LocalBoxFuture, rec::truncate_ranges, split, BaoTree, ChunkNum, ChunkRangesRef, Hash,
        Hasher, TreeNode,
    };

    /// Given a data file and an outboard, compute all valid ranges.
    ///
    /// This is not cheap since it recomputes the hashes for all chunks.
    ///
    /// To reduce the amount of work, you can specify a range you are interested in.
    pub fn valid_ranges<'a, O, D>(
        outboard: O,
        data: D,
        ranges: &'a ChunkRangesRef,
    ) -> impl Stream<Item = io::Result<Range<ChunkNum>>> + 'a
    where
        O: Outboard + 'a,
        D: AsyncSliceReader + 'a,
    {
        Gen::new(move |co| async move {
            if let Err(cause) = RecursiveDataValidator::validate(outboard, data, ranges, &co).await
            {
                co.yield_(Err(cause)).await;
            }
        })
    }

    struct RecursiveDataValidator<'a, O: Outboard, D: AsyncSliceReader> {
        tree: BaoTree,
        shifted_filled_size: TreeNode,
        outboard: O,
        data: D,
        co: &'a Co<io::Result<Range<ChunkNum>>>,
    }

    impl<O: Outboard, D: AsyncSliceReader> RecursiveDataValidator<'_, O, D> {
        async fn validate(
            outboard: O,
            data: D,
            ranges: &ChunkRangesRef,
            co: &Co<io::Result<Range<ChunkNum>>>,
        ) -> io::Result<()> {
            let tree = outboard.tree();
            if tree.blocks() == 1 {
                // special case for a tree that fits in one block / chunk group
                let mut data = data;
                let data = data
                    .read_exact_at(0, tree.size().try_into().unwrap())
                    .await?;
                let actual = O::Hasher::hash_chunk(0, &data, true);
                if actual == outboard.root() {
                    co.yield_(Ok(ChunkNum(0)..tree.chunks())).await;
                }
                return Ok(());
            }
            let ranges = truncate_ranges(ranges, tree.size());
            let root_hash = outboard.root();
            let (shifted_root, shifted_filled_size) = tree.shifted();
            let mut validator = RecursiveDataValidator {
                tree,
                shifted_filled_size,
                outboard,
                data,
                co,
            };
            validator
                .validate_rec(&root_hash, shifted_root, true, ranges)
                .await
        }

        async fn yield_if_valid(
            &mut self,
            range: Range<u64>,
            hash: &Hash,
            is_root: bool,
        ) -> io::Result<()> {
            let len = (range.end - range.start).try_into().unwrap();
            let data = self.data.read_exact_at(range.start, len).await?;
            // is_root is always false because the case of a single chunk group is handled before calling this function
            let actual =
                O::Hasher::hash_chunk(ChunkNum::full_chunks(range.start).0, &data, is_root);
            if &actual == hash {
                // yield the left range
                self.co
                    .yield_(Ok(
                        ChunkNum::full_chunks(range.start)..ChunkNum::chunks(range.end)
                    ))
                    .await;
            }
            io::Result::Ok(())
        }

        fn validate_rec<'b>(
            &'b mut self,
            parent_hash: &'b Hash,
            shifted: TreeNode,
            is_root: bool,
            ranges: &'b ChunkRangesRef,
        ) -> LocalBoxFuture<'b, io::Result<()>> {
            async move {
                if ranges.is_empty() {
                    // this part of the tree is not of interest, so we can skip it
                    return Ok(());
                }
                let node = shifted.subtract_block_size(self.tree.block_size.0);
                let (l, m, r) = self.tree.leaf_byte_ranges3(node);
                if !self.tree.is_relevant_for_outboard(node) {
                    self.yield_if_valid(l..r, parent_hash, is_root).await?;
                    return Ok(());
                }
                let Some((l_hash, r_hash)) = self.outboard.load(node).await? else {
                    // outboard is incomplete, we can't validate
                    return Ok(());
                };
                let actual = O::Hasher::hash_inner(&l_hash, &r_hash, is_root);
                if &actual != parent_hash {
                    // hash mismatch, we can't validate
                    return Ok(());
                };
                let (l_ranges, r_ranges) = split(ranges, node);
                if shifted.is_leaf() {
                    if !l_ranges.is_empty() {
                        self.yield_if_valid(l..m, &l_hash, false).await?;
                    }
                    if !r_ranges.is_empty() {
                        self.yield_if_valid(m..r, &r_hash, false).await?;
                    }
                } else {
                    // recurse (we are in the domain of the shifted tree)
                    let left = shifted.left_child().unwrap();
                    self.validate_rec(&l_hash, left, false, l_ranges).await?;
                    let right = shifted.right_descendant(self.shifted_filled_size).unwrap();
                    self.validate_rec(&r_hash, right, false, r_ranges).await?;
                }
                Ok(())
            }
            .boxed_local()
        }
    }

    /// Given just an outboard, compute all valid ranges.
    ///
    /// This is not cheap since it recomputes the hashes for all chunks.
    pub fn valid_outboard_ranges<'a, O>(
        outboard: O,
        ranges: &'a ChunkRangesRef,
    ) -> impl Stream<Item = io::Result<Range<ChunkNum>>> + 'a
    where
        O: Outboard + 'a,
    {
        Gen::new(move |co| async move {
            if let Err(cause) = RecursiveOutboardValidator::validate(outboard, ranges, &co).await {
                co.yield_(Err(cause)).await;
            }
        })
    }

    struct RecursiveOutboardValidator<'a, O: Outboard> {
        tree: BaoTree,
        shifted_filled_size: TreeNode,
        outboard: O,
        co: &'a Co<io::Result<Range<ChunkNum>>>,
    }

    impl<O: Outboard> RecursiveOutboardValidator<'_, O> {
        async fn validate(
            outboard: O,
            ranges: &ChunkRangesRef,
            co: &Co<io::Result<Range<ChunkNum>>>,
        ) -> io::Result<()> {
            let tree = outboard.tree();
            if tree.blocks() == 1 {
                // special case for a tree that fits in one block / chunk group
                co.yield_(Ok(ChunkNum(0)..tree.chunks())).await;
                return Ok(());
            }
            let ranges = truncate_ranges(ranges, tree.size());
            let root_hash = outboard.root();
            let (shifted_root, shifted_filled_size) = tree.shifted();
            let mut validator = RecursiveOutboardValidator {
                tree,
                shifted_filled_size,
                outboard,
                co,
            };
            validator
                .validate_rec(&root_hash, shifted_root, true, ranges)
                .await
        }

        fn validate_rec<'b>(
            &'b mut self,
            parent_hash: &'b Hash,
            shifted: TreeNode,
            is_root: bool,
            ranges: &'b ChunkRangesRef,
        ) -> LocalBoxFuture<'b, io::Result<()>> {
            Box::pin(async move {
                let yield_node_range = |range: Range<u64>| {
                    self.co.yield_(Ok(
                        ChunkNum::full_chunks(range.start)..ChunkNum::chunks(range.end)
                    ))
                };
                if ranges.is_empty() {
                    // this part of the tree is not of interest, so we can skip it
                    return Ok(());
                }
                let node = shifted.subtract_block_size(self.tree.block_size.0);
                let (l, m, r) = self.tree.leaf_byte_ranges3(node);
                if !self.tree.is_relevant_for_outboard(node) {
                    yield_node_range(l..r).await;
                    return Ok(());
                }
                let Some((l_hash, r_hash)) = self.outboard.load(node).await? else {
                    // outboard is incomplete, we can't validate
                    return Ok(());
                };
                let actual = O::Hasher::hash_inner(&l_hash, &r_hash, is_root);
                if &actual != parent_hash {
                    // hash mismatch, we can't validate
                    return Ok(());
                };
                let (l_ranges, r_ranges) = split(ranges, node);
                if shifted.is_leaf() {
                    if !l_ranges.is_empty() {
                        yield_node_range(l..m).await;
                    }
                    if !r_ranges.is_empty() {
                        yield_node_range(m..r).await;
                    }
                } else {
                    // recurse (we are in the domain of the shifted tree)
                    let left = shifted.left_child().unwrap();
                    self.validate_rec(&l_hash, left, false, l_ranges).await?;
                    let right = shifted.right_descendant(self.shifted_filled_size).unwrap();
                    self.validate_rec(&r_hash, right, false, r_ranges).await?;
                }
                Ok(())
            })
        }
    }
}
#[cfg(feature = "validate")]
pub use validate::{valid_outboard_ranges, valid_ranges};
