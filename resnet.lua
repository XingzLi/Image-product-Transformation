--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

require 'nn'
require 'cunn'
require 'cudnn'

-- local Convolution = cudnn.SpatialConvolution
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel()
   --local depth = opt.depth
   local opt = {}
   local shortcutType = opt.shortcutType or 'C'
   local tensorType = opt.tensorType or 'torch.CudaDoubleTensor'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(convtype, nInputPlane, nOutputPlane, stride)
      --local Convolution
	  --if convtype == 'downsample' then
		 --Convolution = SpatialConvolution
	  --else
		 --Convolution = SpatialFullConvolution
	  --end
	  
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         local Convolution = SpatialConvolution
         local adjpad = 0
         if stride==2 and convtype == 'upsample' then
            adjpad = 1
            Convolution = SpatialFullConvolution
         end
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride,0,0,adjpad,adjpad))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end
    
   local function agent(convtype, nInputPlane, nOutputPlane, stride)
        return nn.Sequential()
            :add(SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
            :add(ReLU(true))
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(convtype, n, stride)
      --local Convolution
      --if convtype == 'downsample' then
          --Convolution = SpatialConvolution
	  --else
          --Convolution = SpatialFullConvolution
	  --end
	  
      local nInputPlane = iChannels
      iChannels = n
      
      local Convolution = SpatialConvolution
      local adjpad = 0
      if stride==2 and convtype == 'upsample' then
            adjpad = 1
            Convolution = SpatialFullConvolution
      end

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1,adjpad,adjpad))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(SpatialConvolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
             :add(nn.ConcatTable()
                :add(s)
                :add(shortcut(convtype, nInputPlane, n, stride)))
             :add(nn.CAddTable(true))
             :add(ReLU(true))
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(convtype, n, stride)
      local Convolution
   	  if convtype == 'downsample' then
		 Convolution = SpatialConvolution
	  else
		 Convolution = SpatialFullConvolution
	  end
	  
      local nInputPlane = iChannels
	  iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
             :add(nn.ConcatTable()
                :add(s)
                :add(shortcut(convtype, nInputPlane, n * 4, stride)))
             :add(nn.CAddTable(true))
             :add(ReLU(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(convtype, block, Infeatures, features, count, stride)
      local s = nn.Sequential()
      iChannels = Infeatures
      for i=1,count do
         s:add(block(convtype, features, i == 1 and stride or 1))
		 --if convtype == 'upsample' and i ~= count then
		    --s:add(ReLU(true))
		 --end
      end
      return s
   end

   local model = nn.Sequential()
   -- Configurations for ResNet:
   --  num. residual blocks, num features, residual block function
   local cfg = {
	 [34]  = {{3, 4, 6, 3}, 512, basicblock},
	 [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
    }

    --assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
	local def1, nFeatures, block1 = table.unpack(cfg[50])
    local def2, nFeatures, block2 = table.unpack(cfg[34])


	-- The ResNet ImageNet model
	model:add(SpatialConvolution(3,64,7,7,1,1,3,3)) --
	model:add(SBatchNorm(64))
	model:add(ReLU(true))
    model:add(layer('downsample', block1, 64, 64, def1[1],2))
    model:add(layer('downsample', block1, 256, 128, def1[2], 2))
    model:add(layer('downsample', block1, 512, 256, def1[3], 2))
    model:add(layer('downsample', block1, 1024, 512, def1[4], 2))
    model:add(layer('upsample', block2, 2048, 512, def2[4], 2))     
    model:add(layer('upsample', block2, 512, 256, def2[3], 2))
    model:add(layer('upsample', block2, 256, 128, def2[2], 2))
    model:add(layer('upsample', block2, 128, 64, def2[1],2))

   
	--local stage1 = nn.Sequential()
         --:add(nn.ConcatTable()
            --:add(nn.Sequential()
               --:add(layer('downsample', block1, 1024, 512, def1[4], 2))
               --:add(layer('upsample', block2, 2048, 512, def2[4], 2)))
            --:add(agent('downsample', 1024, 512, 1)))
         --:add(nn.CAddTable(true))
		 
        
	--local stage2 = nn.Sequential()
         --:add(nn.ConcatTable()
            --:add(nn.Sequential()
				 --:add(layer('downsample', block1, 512, 256, def1[3], 2))
				 --:add(stage1)
				 --:add(layer('upsample', block2, 512, 256, def2[3], 2))
            --:add(agent('downsample', 512, 256, 1)))
         --:add(nn.CAddTable(true))
		
       
	--local stage3 = nn.Sequential()
         --:add(nn.ConcatTable()
            --:add(nn.Sequential()
				 --:add(layer('downsample', block1, 256, 128, def1[2], 2))
				 --:add(stage2)
				 --:add(layer('upsample', block2, 256, 128, def2[2], 2))
            --:add(agent('downsample', 256, 128, 1)))
         --:add(nn.CAddTable(true))
	
    
	--local stage4 = nn.Sequential()
         --:add(nn.ConcatTable()
            --:add(nn.Sequential()
				 --:add(Max(3,3,2,2,1,1))
				 --:add(layer('downsample', block1, 64, 64, def1[1],2))
				 --:add(stage3)
				 --:add(layer('upsample', block2, 128, 64, def2[1],2))
            --:add(agent('downsample', 64, 64, 1)))
         --:add(nn.CAddTable(true))
	
	--model:add(stage4)
	model:add(SpatialConvolution(64,3,7,7,1,1,3,3))
    model:add(nn.Tanh())

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('cudnn.SpatialFullConvolution')
   ConvInit('nn.SpatialConvolution')
   ConvInit('nn.SpatialFullConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
