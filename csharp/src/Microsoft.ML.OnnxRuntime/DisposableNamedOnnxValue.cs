// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntime
{
    public interface IDisposableReadOnlyCollection<T> : IReadOnlyCollection<T>, IDisposable
    {

    }

    internal class DisposableList<T> : List<T>, IDisposableReadOnlyCollection<T>
        where T : IDisposable
    {
        public DisposableList() { }
        public DisposableList(int count) : base(count) { }

        #region IDisposable Support

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                // Dispose in the reverse order.
                // Objects should typically be destroyed/disposed
                // in the reverse order of its creation
                // especially if the objects created later refer to the
                // objects created earlier. For homogeneous collections of objects
                // it would not matter.
                for (int i = this.Count - 1; i >= 0; --i)
                {
                    this[i]?.Dispose();
                }
                this.Clear();
            }
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }

    /// <summary>
    /// This class owns an OrtValue that 
    /// </summary>
    public class DisposableNamedOnnxValue : NamedOnnxValue, IDisposable
    {
        private IOrtValueOwner _ortValueHolder;
        private TensorElementType _elementType;
        private OnnxValueType _onnxValueType;
        private bool _disposed = false;

        private DisposableNamedOnnxValue(string name, Object value, OnnxValueType onnxValueType, TensorElementType elementType, IOrtValueOwner ortValueHolder)
            : base(name, value)
        {
            _onnxValueType = onnxValueType;
            _elementType = elementType;
            _ortValueHolder = ortValueHolder;
        }

        /// <summary>
        /// Overrides the base class method. Since the instance already owns underlying OrtValue handle,
        /// it returns an instance of OrtValue that does not own the raw handle
        /// that to the output onnxValue. With respect to pinnedMemoryHandle, it has no operation
        /// to do, as this class maintains a managed buffer via _nativeMememoryManager and the memory will be
        /// disposed by it. This is the case when we are dealing with an OrtValue that is backed by native memory
        /// and not by pinned managed memory
        /// </summary>
        /// <param name="pinnedMemoryHandle">always set to null</param>
        /// <returns>An instance of OrtValue that does not own underlying memory</returns>
        internal override OrtValue ToOrtValue(out MemoryHandle? pinnedMemoryHandle)
        {
            // PinnedMemoryHandle holds the default value as DisposableNamedOnnxValue
            // doesn't hold any managed buffer (that needs to be pinned)
            pinnedMemoryHandle = null;
            // Return non-owning instance of OrtValue
            return _ortValueHolder.Value;
        }

        /// <summary>
        /// Creates an instance of DisposableNamedOnnxValue and takes ownership of ortValue
        /// on success.
        /// </summary>
        /// <param name="name">name of the value</param>
        /// <param name="ortValue">underlying OrtValue</param>
        /// <returns></returns>
        internal static DisposableNamedOnnxValue CreateTensorFromOnnxValue(string name, OrtValue ortValue)
        {
            DisposableNamedOnnxValue result = null;

            /* Get Tensor element type */  //TODO: Assumed value is Tensor, need to support non-tensor types in future
            IntPtr typeAndShape = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorTypeAndShape(ortValue.Handle, out typeAndShape));
            TensorElementType elemType = TensorElementType.DataTypeMax;
            try
            {
                IntPtr el_type;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(typeAndShape, out el_type));
                elemType = (TensorElementType)el_type;
            }
            finally
            {
                NativeMethods.OrtReleaseTensorTypeAndShapeInfo(typeAndShape);
            }

            switch (elemType)
            {
                case TensorElementType.Float:
                    result = DisposableNamedOnnxValueFromNativeTensor<float>(name, ortValue);
                    break;
                case TensorElementType.Double:
                    result = DisposableNamedOnnxValueFromNativeTensor<double>(name, ortValue);
                    break;
                case TensorElementType.Int16:
                    result = DisposableNamedOnnxValueFromNativeTensor<short>(name, ortValue);
                    break;
                case TensorElementType.UInt16:
                    result = DisposableNamedOnnxValueFromNativeTensor<ushort>(name, ortValue);
                    break;
                case TensorElementType.Int32:
                    result = DisposableNamedOnnxValueFromNativeTensor<int>(name, ortValue);
                    break;
                case TensorElementType.UInt32:
                    result = DisposableNamedOnnxValueFromNativeTensor<uint>(name, ortValue);
                    break;
                case TensorElementType.Int64:
                    result = DisposableNamedOnnxValueFromNativeTensor<long>(name, ortValue);
                    break;
                case TensorElementType.UInt64:
                    result = DisposableNamedOnnxValueFromNativeTensor<ulong>(name, ortValue);
                    break;
                case TensorElementType.UInt8:
                    result = DisposableNamedOnnxValueFromNativeTensor<byte>(name, ortValue);
                    break;
                case TensorElementType.Int8:
                    result = DisposableNamedOnnxValueFromNativeTensor<sbyte>(name, ortValue);
                    break;
                case TensorElementType.String:
                    result = DisposableNamedOnnxValueFromNativeTensor<string>(name, ortValue);
                    break;
                case TensorElementType.Bool:
                    result = DisposableNamedOnnxValueFromNativeTensor<bool>(name, ortValue);
                    break;
                default:
                    throw new NotSupportedException("Tensor of element type: " + elemType + " is not supported");

            }

            return result;
        }

        internal static DisposableNamedOnnxValue CreateSequenceFromOrtValue(string name, OrtValue ortValue, OrtAllocator allocator)
        {
            IntPtr count = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValueCount(ortValue.Handle, out count));
            // Sequence is going to be owned by NameOnnxValue and it does not dispose anything anyway
            var sequence = new List<DisposableNamedOnnxValue>(count.ToInt32());
            for (int i = 0; i < count.ToInt32(); i++)
            {
                IntPtr nativeOnnxValueSeq;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValue(ortValue.Handle, i, allocator.Pointer, out nativeOnnxValueSeq));
                sequence.Add(CreateFromOrtValue(string.Empty, new OrtValue(nativeOnnxValueSeq), allocator));
            }
            return new DisposableNamedOnnxValue(name, sequence, OnnxValueType.ONNX_TYPE_SEQUENCE, TensorElementType.DataTypeMax, ortValue);
        }

        internal static DisposableNamedOnnxValue CreateFromOrtValue(string name, OrtValue ortValue)
        {
            return CreateFromOrtValue(name, ortValue, OrtAllocator.DefaultInstance);
        }

        internal static DisposableNamedOnnxValue CreateFromOrtValue(string name, OrtValue ortValue, OrtAllocator allocator)
        {
            IntPtr valueType;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValueType(ortValue.Handle, out valueType));
            OnnxValueType onnxValueType = (OnnxValueType)valueType;
            switch (onnxValueType)
            {
                case OnnxValueType.ONNX_TYPE_TENSOR:
                    return CreateTensorFromOnnxValue(name, ortValue);

                case OnnxValueType.ONNX_TYPE_SEQUENCE:
                    IntPtr count = IntPtr.Zero;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValueCount(ortValue.Handle, out count));
                    var sequence = new DisposableList<DisposableNamedOnnxValue>(count.ToInt32());
                    for (int i = 0; i < count.ToInt32(); i++)
                    {
                        IntPtr nativeOnnxValueSeq;
                        NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValue(ortValue.Handle, i, allocator.Pointer, out nativeOnnxValueSeq));
                        sequence.Add(CreateFromOrtValue(string.Empty, new OrtValue(nativeOnnxValueSeq), allocator));
                    }
                    return new DisposableNamedOnnxValue(name, sequence, OnnxValueType.ONNX_TYPE_SEQUENCE, TensorElementType.DataTypeMax, null);

                case OnnxValueType.ONNX_TYPE_MAP:
                    IntPtr nativeOnnxValueMapKeys = IntPtr.Zero;
                    IntPtr nativeOnnxValueMapValues = IntPtr.Zero;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValue(ortValue.Handle, 0, allocator.Pointer, out nativeOnnxValueMapKeys));
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValue(ortValue.Handle, 1, allocator.Pointer, out nativeOnnxValueMapValues));

                    IntPtr typeAndShape = IntPtr.Zero;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorTypeAndShape(nativeOnnxValueMapKeys, out typeAndShape));
                    TensorElementType elemType = TensorElementType.DataTypeMax;
                    try 
                    {
                        IntPtr el_type;
                        NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(typeAndShape, out el_type));
                        elemType = (TensorElementType)el_type;
                    }
                    finally
                    {
                        NativeMethods.OrtReleaseTensorTypeAndShapeInfo(typeAndShape);
                    }

                    switch (elemType)
                    {
                        case TensorElementType.Int64:
                            return DisposableNamedOnnxValueFromNativeMap<Int64, float>(string.Empty, nativeOnnxValueMapKeys, nativeOnnxValueMapValues);
                        case TensorElementType.String:
                            return DisposableNamedOnnxValueFromNativeMap<string, float>(string.Empty, nativeOnnxValueMapKeys, nativeOnnxValueMapValues);
                        default:
                            throw new NotSupportedException("Map of element type: " + elemType + " is not supported");
                    }
                default:
                    throw new NotSupportedException("OnnxValueType : " + onnxValueType + " is not supported");
            }
        }

        private static DisposableNamedOnnxValue DisposableNamedOnnxValueFromNativeTensor<T>(string name, OrtValue ortValue)
        {
            if (typeof(T) == typeof(string))
            {
                var nativeTensorWrapper = new NativeOnnxTensorMemory<string>(ortValue);
                var dt = new DenseTensor<string>(nativeTensorWrapper.GetBytesAsStringMemory(), nativeTensorWrapper.Dimensions);
                return new DisposableNamedOnnxValue(name, dt, OnnxValueType.ONNX_TYPE_TENSOR, nativeTensorWrapper.ElementType, nativeTensorWrapper);
            }
            else
            {
                NativeOnnxTensorMemory<T> nativeTensorWrapper = new NativeOnnxTensorMemory<T>(ortValue);
                DenseTensor<T> dt = new DenseTensor<T>(nativeTensorWrapper.Memory, nativeTensorWrapper.Dimensions);
                return new DisposableNamedOnnxValue(name, dt, OnnxValueType.ONNX_TYPE_TENSOR, nativeTensorWrapper.ElementType, nativeTensorWrapper);
            }
        }

        private static DisposableNamedOnnxValue DisposableNamedOnnxValueFromNativeMap<K, V>(string name, IntPtr nativeOnnxValueKeys, IntPtr nativeOnnxValueValues)
        {
            var nativeTensorWrapperValues = new NativeOnnxTensorMemory<V>(nativeOnnxValueValues);
            var denseTensorValues = new DenseTensor<V>(nativeTensorWrapperValues.Memory, nativeTensorWrapperValues.Dimensions);

            if (typeof(K) == typeof(string))
            {
                var map = new Dictionary<string, V>();
                var nativeTensorWrapper = new NativeOnnxTensorMemory<string>(nativeOnnxValueKeys);
                var denseTensorKeys = new DenseTensor<string>(nativeTensorWrapper.GetBytesAsStringMemory(), nativeTensorWrapper.Dimensions);
                for (var i = 0; i < denseTensorKeys.Length; i++)
                {
                    map.Add(denseTensorKeys.GetValue(i), denseTensorValues.GetValue(i));
                }
                // release native memory
                nativeTensorWrapperValues.Dispose();
                nativeTensorWrapper.Dispose();
                return new DisposableNamedOnnxValue(string.Empty, map, OnnxValueType.ONNX_TYPE_MAP, TensorElementType.DataTypeMax, null);
            }
            else
            {
                var map = new Dictionary<K, V>();
                var nativeTensorWrapper = new NativeOnnxTensorMemory<K>(nativeOnnxValueKeys);
                var denseTensorKeys = new DenseTensor<K>(nativeTensorWrapper.Memory, nativeTensorWrapper.Dimensions);
                for (var i = 0; i < denseTensorKeys.Length; i++)
                {
                    map.Add(denseTensorKeys.GetValue(i), denseTensorValues.GetValue(i));
                }
                // release native memory
                nativeTensorWrapperValues.Dispose();
                nativeTensorWrapper.Dispose();
                return new DisposableNamedOnnxValue(string.Empty, map, OnnxValueType.ONNX_TYPE_MAP, TensorElementType.DataTypeMax, null);
            }
        }

        #region IDisposable Support

        protected virtual void Dispose(bool disposing)
        {
            if(_disposed)
            {
                return;
            }

            // dispose managed state (managed objects).
            if (disposing)
            {
                if (_ortValueHolder != null)
                {
                    _ortValueHolder.Dispose();
                    _ortValueHolder = null;
                }
            }
            _disposed = true;
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion

    }
}
