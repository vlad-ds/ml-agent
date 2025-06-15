"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { FileSpreadsheet, Upload } from "lucide-react"
import { useState } from "react"
import { useRouter } from "next/navigation"

export default function UploadPage() {
  const [files, setFiles] = useState<File[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadDirectory, setUploadDirectory] = useState<string | null>(null)
  const router = useRouter()

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFiles(Array.from(event.target.files))
      setUploadDirectory(null)
    }
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    if (event.dataTransfer.files) {
      setFiles(Array.from(event.dataTransfer.files))
      setUploadDirectory(null)
    }
  }

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
  }

  const handleUpload = async () => {
    if (files.length === 0) return

    setIsUploading(true)
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const data = await response.json()
      setUploadDirectory(data.directory)
      console.log('Files uploaded successfully')
      
      // Redirect to analysis page after successful upload
      router.push(`/analyze/${data.directory}`)
    } catch (error) {
      console.error('Error uploading files:', error)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="flex flex-col min-h-screen justify-center items-center">
    <div className="container max-w-4xl py-10">
      <div className="space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-3xl font-bold">Upload your dataset</h1>
          <p className="text-muted-foreground">
            Drop your CSV or Excel files to start the automatic analysis
          </p>
        </div>

        <Card className="border-dashed">
          <CardHeader>
            <CardTitle>Drop your files</CardTitle>
            <CardDescription>
              Supported formats: CSV, Excel (.xlsx, .xls)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div 
              className="flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-lg bg-muted/50"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <FileSpreadsheet className="h-12 w-12 text-muted-foreground mb-4" />
              <div className="text-center space-y-2">
                <p className="text-sm text-muted-foreground">
                  Drag and drop your files here, or
                </p>
                <input
                  type="file"
                  multiple
                  accept=".csv,.xlsx,.xls"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <Button 
                  variant="outline"
                  onClick={() => document.getElementById('file-upload')?.click()}
                >
                  <Upload className="mr-2 h-4 w-4" />
                  Browse files
                </Button>
                {files.length > 0 && (
                  <div className="mt-4">
                    <p className="text-sm font-medium">Selected files:</p>
                    <ul className="text-sm text-muted-foreground">
                      {files.map((file, index) => (
                        <li key={index}>{file.name}</li>
                      ))}
                    </ul>
                    <Button 
                      onClick={handleUpload}
                      disabled={isUploading}
                      className="mt-4"
                    >
                      {isUploading ? 'Uploading...' : 'Upload Files'}
                    </Button>
                  </div>
                )}
                {uploadDirectory && (
                  <div className="mt-4 p-4 bg-green-50 rounded-lg">
                    <p className="text-sm font-medium text-green-800">Upload successful!</p>
                    <p className="text-sm text-green-600">Directory: {uploadDirectory}</p>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Instructions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <h3 className="font-medium">Data Preparation</h3>
              <ul className="list-disc list-inside text-muted-foreground space-y-1">
                <li>Make sure your file contains column headers</li>
                <li>Data should be cleaned and properly formatted</li>
                <li>Avoid special characters in column names</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h3 className="font-medium">Limits</h3>
              <ul className="list-disc list-inside text-muted-foreground space-y-1">
                <li>Maximum file size: 100 MB</li>
                <li>Maximum number of columns: 100</li>
                <li>Maximum number of rows: 1,000,000</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
    </div>
  )
} 