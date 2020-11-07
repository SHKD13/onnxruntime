// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Provides access from the underlying object that owns disposable OrtValue
    /// The returned value does not own the actual memory and does nothing on Dispose()
    /// </summary>
    internal interface IOrtValueOwner : IDisposable
    {
        OrtValue Value { get; }
    }

    /// <summary>
    /// This helper class owns the underlying OrtValue that is assumed to be a Tensor,
    /// it does not support any other ortValues and caches Tensor properties.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class NativeOnnxTensorMemory<T> : MemoryManager<T>, IOrtValueOwner
    {
        private bool _disposed = false;
        private OrtValue _ortValue; // Disposable
        private IntPtr _dataBufferPointer;    // pointer to mutable tensor data in native memory
        private string[] _dataBufferAsString; // string tensor values copied into managed memory
        private Tensors.TensorElementType _elementType;
        private int _elementCount;
        private int _elementWidth;
        private int[] _dimensions;

        /// <summary>
        /// Constructs an instance and takes ownership of ortValue on success
        /// </summary>
        /// <param name="ortValue">ortValue that is a Tensor</param>
        public NativeOnnxTensorMemory(OrtValue ortValue)
        {
            Type type = null;
            int width = 0;
            IntPtr typeAndShape = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorTypeAndShape(ortValue.Handle, out typeAndShape));
            try
            {
                TensorElementType elemType;
                {
                    IntPtr el_type;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(typeAndShape, out el_type));
                    elemType = (TensorElementType)el_type;
                }
                TensorElementTypeConverter.GetTypeAndWidth(elemType, out type, out width);

                if (typeof(T) != type)
                    throw new NotSupportedException(nameof(NativeOnnxTensorMemory<T>) + " does not support T = " + nameof(T));

                _elementType = elemType;
                _elementWidth = width;
                UIntPtr dimension;
                long count;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(typeAndShape, out dimension));
                {
                    IntPtr el_count;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorShapeElementCount(typeAndShape, out el_count));  // count can be negative. 
                    count = (long)el_count;
                }
                if (count < 0)
                {
                    throw new NotSupportedException("Symbolic dimensions in the tensor is not supported");
                }

                long[] shape = new long[dimension.ToUInt64()];
                 NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensions(typeAndShape, shape, dimension)); //Note: shape must be alive during the call

                _elementCount = (int)count;
                _dimensions = new int[dimension.ToUInt64()];
                for (ulong i = 0; i < dimension.ToUInt64(); i++)
                {
                    _dimensions[i] = (int)shape[i];
                }

                if (typeof(T) != typeof(string))
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorMutableData(ortValue.Handle, out _dataBufferPointer));
                }
                else
                {
                    UIntPtr strLen;
                    var offsets = new UIntPtr[_elementCount];
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetStringTensorDataLength(_ortValue.Handle, out strLen));
                    var dataBuffer = new byte[strLen.ToUInt64()];

                    using (var dataBufferHandle = new Memory<byte>(dataBuffer).Pin())
                    using (var offsetMemoryHandle = new Memory<UIntPtr>(offsets).Pin())
                    {
                        unsafe
                        {
                            _dataBufferPointer = (IntPtr)dataBufferHandle.Pointer;
                            NativeApiStatus.VerifySuccess(
                                NativeMethods.OrtGetStringTensorContent(
                                ortValue.Handle, _dataBufferPointer, strLen,
                                (IntPtr)offsetMemoryHandle.Pointer,
                                (UIntPtr)_elementCount));
                        }
                        _dataBufferAsString = new string[_elementCount];

                        for (var i = 0; i < offsets.Length; i++)
                        {
                            var length = (i == offsets.Length - 1)
                                ? strLen.ToUInt64() - offsets[i].ToUInt64()
                                : offsets[i + 1].ToUInt64() - offsets[i].ToUInt64();
                            // Onnx specifies strings always in UTF-8, no trailing null, no leading BOM
                            _dataBufferAsString[i] = Encoding.UTF8.GetString(dataBuffer, (int)offsets[i], (int)length);
                        }
                    }
                }
                // Transfer ownership
                _ortValue = new OrtValue(ortValue.Disown());
            }
            finally
            {
                NativeMethods.OrtReleaseTensorTypeAndShapeInfo(typeAndShape);
            }
        }

        /// <summary>
        /// Returns a non-owning copy of OrtValue so the
        /// result can not release native memory
        /// </summary>
        public OrtValue Value { get { return new OrtValue(_ortValue.Handle, false); } }

        public bool IsDisposed => _disposed;

        public int[] Dimensions => _dimensions;

        public int Rank => _dimensions.Length;

        public int Count => _elementCount;

        public int ElementWidth => _elementWidth;

        public Tensors.TensorElementType ElementType => _elementType;

        public override Span<T> GetSpan()
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(NativeOnnxTensorMemory<T>));
            Span<T> span = null;
            unsafe
            {
                span = new Span<T>((void*)_dataBufferPointer, _elementCount);
            }

            return span;
        }

        public Memory<String> GetBytesAsStringMemory()
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(NativeOnnxTensorMemory<T>));

            if (typeof(T) != typeof(string))
                throw new NotSupportedException(nameof(NativeOnnxTensorMemory<T>.GetBytesAsStringMemory) + ": T must be byte");

            return (_dataBufferAsString == null) ? new Memory<string>() : new Memory<string>(_dataBufferAsString);
        }

        public override MemoryHandle Pin(int elementIndex = 0)
        {
            //Note: always pin the full buffer and return
            unsafe
            {
                if (elementIndex >= _elementCount)
                {
                    throw new ArgumentOutOfRangeException(nameof(elementIndex));
                }
                return new MemoryHandle((void*)((int)_dataBufferPointer + elementIndex * _elementWidth)); //could not use Unsafe.Add
            }
        }

        // MemoryHandle returned above by Pin() should be disposed.
        // Unpin() is purely to satisfy the interface.
        // TODO: This class needs work. It is not clear what happens
        // if the MemoryHandle remains alive and this class gets Disposed.
        public override void Unpin() { }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected override void Dispose(bool disposing)
        {
            if(_disposed)
            {
                return;
            }

            if (_ortValue != null)
            {
                _ortValue.Dispose();
                _ortValue = null;
            }
            _disposed = true;
        }

        protected override bool TryGetArray(out ArraySegment<T> arraySegment)
        {
            // cannot expose managed array
            arraySegment = default(ArraySegment<T>);
            return false;
        }
    }
}
