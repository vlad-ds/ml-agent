import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"
import Link from "next/link"
import { WarpBackground } from "@/components/magicui/warp-background";

export default function Home() {
  return (
    <WarpBackground className="min-h-screen flex flex-col items-center justify-center">
    <div className="flex flex-col h-full w-full items-center justify-center">
      {/* Hero Section */}
      <section className="flex-1 bg-white h-fit rounded-lg border flex flex-col w-fit px-12 py-12 items-center justify-center text-center space-y-8 relative">
        
        <div className="space-y-8 relative z-10">
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
            Analyze your data with AI
            <span className="text-primary"> in one click</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Our AI agent analyzes your datasets and provides relevant insights. No more need to be an expert in data science.
          </p>
          <div className="flex gap-4 justify-center">
            <Button size="lg" asChild>
              <Link href="/analyze">
                Start now <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
            <Button size="lg" variant="outline">
              Learn more
            </Button>
          </div>
        </div>
      </section>
    </div>
    </WarpBackground>
  )
}
